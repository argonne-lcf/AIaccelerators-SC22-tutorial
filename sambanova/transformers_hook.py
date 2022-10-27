import argparse
import os.path
import sys
from typing import List

import torch
import torch.nn as nn
import yaml
from tasks.utils.common_utils import loss_scale_of_crossentropy
from transformers import set_seed
from transformers_base import build_arguments, get_dataset, get_task_module
from transformers_run import preconstruct_from_yaml

import sambaflow.samba.utils as sn_utils
from sambaflow import is_demo, samba
from sambaflow.samba.utils import assert_close
from sambaflow.samba.utils import set_seed as set_samba_seed
from sambaflow.samba.utils.argparser import parse_yaml_to_args
from sambaflow.samba.utils.pef_utils import get_pefmeta

HOST_MEMORY = {'host': True, 'device': False, 'auto': None}


def run_torch(model: nn.Module, inputs: List[torch.Tensor], inference_only: bool) -> torch.Tensor:
    inputs = inputs[:-1] + (samba.to_torch(inputs[-1]).long(), )
    if inference_only:
        with torch.no_grad():
            return model(*inputs)
    else:
        outputs = model(*inputs)
        outputs[0].backward()
        return outputs


def test(args: argparse.Namespace, model: nn.Module, inputs: List[samba.SambaTensor],
         outputs: List[samba.SambaTensor]) -> None:
    samba_inputs = [input for input in inputs if input is not None]
    gold_inputs = [samba.to_torch(input) for input in samba_inputs]
    gold_inputs = [gold_inputs[0], gold_inputs[1], gold_inputs[2], gold_inputs[3], None, None, gold_inputs[4]]

    # We use this to implement ignored_index in CE, HF uses default -100 as ignored_index
    grad_scale = [loss_scale_of_crossentropy(inputs[-1], -100), None] if args.task_name == 'mlm' else None
    # Run RDU version of model to ensure no hang
    samba.session.run(input_tensors=samba_inputs,
                      output_tensors=outputs,
                      grad_of_outputs=grad_scale,
                      data_parallel=args.data_parallel,
                      data_parallel_mode=args.data_parallel_mode)
    samba_loss, samba_logits = samba.session.get_tensors(outputs)
    samba_loss *= grad_scale[0] if args.task_name == 'mlm' else 1
    samba_loss = samba_loss.sum()

    t, rtol = args.absolute_numeric_threshold, args.relative_numeric_threshold
    if t >= 0 and rtol >= 0:
        gold_loss, gold_logits = run_torch(model, inputs, args.inference)
        assert_close(samba_loss, gold_loss, 'loss', t, rtol, visualize=args.visualize)
        # TODO(tianxiaoj): add logits assertion later, currently the average difference is more than 50%


def main(argv: List[str], **kwargs):
    # parse yaml file from launch_app.py
    yaml_flag = argv[0]
    if yaml_flag == '--yaml-config':
        yaml_file = argv[1]
        argv = argv[2:]
        try:
            with open(yaml_file, 'r') as settings_file:
                settings = yaml.safe_load(settings_file)
        except:
            raise Exception(f'An error occurred while reading {yaml_file} as yaml config')
        argv, file_to_close = preconstruct_from_yaml(settings, argv)
    else:
        yaml_file = None
    args = build_arguments(argv, yaml_file=yaml_file)

    if is_demo():
        if args.tokenizer_name != "bert-large-uncased":
            print(f"Unsupported tokenizer_name for the demo compiler: {args.tokenizer_name}")
            sys.exit(1)
        if args.task_name != "squad":
            print(f"Unsupported task_name for the demo compiler: {args.task_name}")
            sys.exit(1)

        if args.command == "compile":
            # Set some options to optimize compile time.
            # The end user is responsible for providing these options, normally,
            # but we do it automatically here to streamline the demo experience
            args.mac_v1 = False
            args.disable_retry_lower_visible_resources = True
            args.resources_scaling_factors = ['0.5', '0.5', '0.5']

    # Set Samba Seeds
    set_seed(args.seed)
    set_samba_seed(args.seed)

    task_module = get_task_module(args)
    if args.hook_v2:
        dataset = get_dataset(args)

    # Pull the HuggingFace Configuration
    config = task_module.get_model_config(args)
    model = task_module.get_model(args, config)
    model = task_module.apply_patch(args, model)

    samba.from_torch_(model)
    tokenizer = task_module.get_tokenizer(args)

    optim = task_module.get_optimizers(args, model)

    if not args.use_pretrained_weights is None:
        task_module.load_pretrained_weights(args, model, optim, args.use_pretrained_weights)

    if args.dev_compile:
        train_dataset, eval_dataset, test_dataset = None, None, None
    elif args.hook_v2:
        train_dataset, eval_dataset, test_dataset = [
            dataset(args, mode, tokenizer, args.dataset) for mode in ('train', 'dev', 'test')
        ]
    else:
        train_dataset, eval_dataset, test_dataset = task_module.get_all_datasets(args, tokenizer)

    tracing_dataset = test_dataset if args.do_predict and not (args.do_train or args.do_eval) else train_dataset

    collator = task_module.get_data_collator(args, tokenizer)

    init_output_grads = not (args.ignore_output_grads or args.inference)

    if args.hook_v2:
        traced_inputs = task_module.get_precompile_tracing_inputs(args, collator, tokenizer)
    else:
        traced_inputs = task_module.get_precompile_tracing_inputs(args, tracing_dataset, collator, tokenizer)

    if args.command == "compile":
        config_dict = vars(args)
        if args.generate_mapping_decisions:
            assert args.mac_human_decision is None,\
                "--generate_mapping_decisions was specified, but a value was also provided for --mac-human-decision." \
                " These are mutually exclusive options"
            generated_hd_json = task_module.generate_decisions(config, args.n_chips,
                                                               os.path.join(args.output_folder, args.pef_name), args)
            config_dict['mac_human_decision'] = generated_hd_json

        samba.session.compile(model,
                              traced_inputs,
                              optim,
                              name='hf_transformer',
                              init_output_grads=init_output_grads,
                              squeeze_bs_dim=True,
                              is_split_batch=args.grad_accumulation_steps > 1,
                              graph_transform_hook=task_module.modify_graph_intermediate_representation,
                              app_dir=sn_utils.get_file_dir(__file__),
                              config_dict=config_dict,
                              pef_metadata=get_pefmeta(args, model),
                              io_host_memory=HOST_MEMORY[args.io_memory_location])
    elif args.command == "measure-performance":
        traced_inputs = traced_inputs.values() if isinstance(traced_inputs, dict) else traced_inputs
        traced_outputs = sn_utils.trace_graph(model,
                                              traced_inputs,
                                              optim,
                                              pef=args.pef,
                                              mapping=args.mapping,
                                              distlearn_config=args.distlearn_config,
                                              init_output_grads=False)
        traced_inputs = [input for input in traced_inputs if input is not None]

        throughput, latency = sn_utils.measure_performance(model,
                                                           traced_inputs,
                                                           args.batch_size,
                                                           args.n_chips,
                                                           args.inference,
                                                           run_graph_only=args.run_graph_only,
                                                           n_iterations=args.num_iterations,
                                                           json=args.bench_report_json,
                                                           compiled_stats_json=args.compiled_stats_json,
                                                           data_parallel=args.data_parallel,
                                                           reduce_on_rdu=args.reduce_on_rdu,
                                                           use_sambaloader=False,
                                                           min_duration=args.min_duration,
                                                           set_output_grads=False)
        assert throughput > args.min_throughput, \
            f'Expected throughput to be at least {args.min_throughput}, instead found {throughput}'
    elif args.command == "measure-sections":
        traced_inputs = traced_inputs.values() if isinstance(traced_inputs, dict) else traced_inputs
        _ = sn_utils.trace_graph(model,
                                 traced_inputs,
                                 optim,
                                 pef=args.pef,
                                 mapping=args.mapping,
                                 distlearn_config=args.distlearn_config)
        traced_inputs = [input for input in traced_inputs if input is not None]

        sn_utils.measure_sections(model,
                                  traced_inputs,
                                  num_sections=args.num_sections,
                                  n_iterations=args.num_iterations,
                                  batch_size=args.batch_size,
                                  json=args.bench_report_json,
                                  data_parallel=args.data_parallel,
                                  reduce_on_rdu=args.reduce_on_rdu,
                                  min_duration=args.min_duration)
    elif args.command == "test":
        if args.grad_accumulation_steps == 1:
            ## Trace the graph for running this on chip
            traced_outputs = sn_utils.trace_graph(model,
                                                  traced_inputs,
                                                  optim,
                                                  pef=args.pef,
                                                  mapping=args.mapping,
                                                  distlearn_config=args.distlearn_config,
                                                  init_output_grads=init_output_grads)
            if args.do_generate:
                task_module.test(args, model, traced_inputs, traced_outputs)
            else:
                test(args, model, traced_inputs, traced_outputs)

    elif args.command == "run":
        if hasattr(args, 'fix_rank_rdu_mapping') and args.fix_rank_rdu_mapping:
            sn_utils.fix_rdu_affinity()

        traced_outputs = None
        if not args.train_torch:
            if args.module_name == 'gpt3_patterntest':
                traced_outputs = sn_utils.trace_graph(model,
                                                      traced_inputs,
                                                      optim,
                                                      pef=args.pef,
                                                      mapping=args.mapping,
                                                      dev_mode=True,
                                                      distlearn_config=args.distlearn_config,
                                                      data_parallel_mode=args.data_parallel_mode,
                                                      init_output_grads=init_output_grads)
            else:
                traced_outputs = sn_utils.trace_graph(model,
                                                      traced_inputs,
                                                      optim,
                                                      pef=args.pef,
                                                      mapping=args.mapping,
                                                      distlearn_config=args.distlearn_config,
                                                      data_parallel_mode=args.data_parallel_mode,
                                                      init_output_grads=init_output_grads)

        if args.do_train or args.do_eval or args.do_predict:
            task_module.do_training(args=args,
                                    task_module=task_module,
                                    model=model,
                                    optims=optim,
                                    model_config=config,
                                    collator=collator,
                                    train_dataset=train_dataset,
                                    eval_dataset=eval_dataset,
                                    test_dataset=test_dataset,
                                    tokenizer=tokenizer,
                                    traced_outputs=traced_outputs,
                                    **kwargs)

        elif args.listen_for_input:
            task_module.do_online_inference(args=args, model=model, tokenizer=tokenizer, traced_outputs=traced_outputs)
        elif args.do_generate:
            task_module.generate_text(args, model, tokenizer, traced_outputs)
            return
        elif args.do_test:
            task_module.do_test(args=args,
                                task_module=task_module,
                                model=model,
                                optims=optim,
                                model_config=config,
                                collator=collator,
                                train_dataset=train_dataset,
                                eval_dataset=eval_dataset,
                                test_dataset=test_dataset,
                                tokenizer=tokenizer,
                                traced_outputs=traced_outputs,
                                **kwargs)
            return
        else:
            print("No action specified. Please specify --do_train, --do_eval, or --do_predict")
            return
    else:
        print(f"Command {args.command} not implemented!")
        raise NotImplementedError


if __name__ == "__main__":
    main(sys.argv[1:])
