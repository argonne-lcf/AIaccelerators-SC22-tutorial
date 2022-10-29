#! /bin/bash
set -e
#######################
# Edit these variables.
#######################
export OMP_NUM_THREADS=1
MODEL_NAME="bertinftrn"
#######################
# Start script timer
SECONDS=0
# Temp file location
DIRECTORY=$$
OUTDIR=${HOME}/models_dir/${DIRECTORY}

source /opt/sambaflow/venv/bin/activate
cd ${HOME}
echo "Model: ${MODEL_NAME}"
echo "Date: " $(date +%m/%d/%y)
echo "Time: " $(date +%H:%M)
mkdir ${OUTDIR}
ln -s /usr/local/share/data/bert/SQUAD_DIR/pre-data ${OUTDIR}/
cd ${OUTDIR}
echo "Machine State Before: "
/opt/sambaflow/bin/snfadm -l inventory

echo "===== TRAIN COMPILE" 
COMMAND="python /opt/sambaflow/apps/nlp/transformers_on_rdu/transformers_hook.py compile --tokenizer_name bert-large-uncased --model_name_or_path bert-large-uncased --do_eval --data_dir /usr/local/share/data/bert/SQUAD_DIR/ --max_seq_length 384 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 -b 32 --output_dir=${OUTDIR}/hf_output_squad_compile --overwrite_output_dir --seed 1206287 --task_name squad --module_name squad --mac-human-decision /opt/sambaflow/apps/nlp/transformers_on_rdu/human_decisions/compiler_configs/faster_compile_higher_utilization.json --mac-v2 --cache_dir ${OUTDIR}/squad_cache --pef transformers_hook --output-folder=${OUTDIR}"
echo "===== COMPILE COMMAND: $COMMAND"
eval $COMMAND

echo "===== TRAIN RUN"
COMMAND="python /opt/sambaflow/apps/nlp/transformers_on_rdu/transformers_hook.py run --model_name_or_path bert-large-uncased --tokenizer_name bert-large-uncased --do_train --do_eval --data_dir /usr/local/share/data/bert/SQUAD_DIR/ -p ${OUTDIR}/transformers_hook/transformers_hook.pef --max_seq_length 384 --per_device_train_batch_size 32 --per_device_eval_batch_size 32 -b 32 --output_dir=${OUTDIR}/hf_output_squad_run  --overwrite_output_dir --seed 1206287 --task_name squad --module_name squad --learning_rate 3e-05  --eval_steps 1000 --num_train_epochs 0.2 --cache_dir ${OUTDIR}/squad_cache"

echo "===== RUN COMMAND: $COMMAND"
eval $COMMAND

echo "===== INF COMPILE"
COMMAND="python /opt/sambaflow/apps/nlp/transformers_on_rdu/transformers_hook.py compile --inference --model_name_or_path bert-large-uncased --tokenizer_name bert-large-uncased --do_eval --data_dir /usr/local/share/data/bert/SQUAD_DIR/ --max_seq_length 384 --per_device_eval_batch_size 32 -b 32 --output_dir=${OUTDIR}/hf_output_squad_inference_compile --overwrite_output_dir --seed 1206287 --task_name squad --module_name squad --mac-human-decision /opt/sambaflow/apps/nlp/transformers_on_rdu/human_decisions/compiler_configs/faster_compile_higher_utilization.json --mac-v2 --cache_dir ${OUTDIR}/squad_cache --pef transformers_hook_inf --output-folder=${OUTDIR}"
echo "===== INF COMPILE COMMAND: $COMMAND"
eval $COMMAND

echo "===== INF RUN"
COMMAND="python /opt/sambaflow/apps/nlp/transformers_on_rdu/transformers_hook.py run --inference --model_name_or_path ${OUTDIR}/hf_output_squad_run --do_eval --data_dir /usr/local/share/data/bert/SQUAD_DIR/ -p ${OUTDIR}/transformers_hook_inf/transformers_hook_inf.pef --max_seq_length 384 --per_device_eval_batch_size 32 -b 32 --output_dir=${OUTDIR}/hf_output_squad_inference --overwrite_output_dir --seed 1206287 --task_name squad --module_name squad --learning_rate 3e-05  --eval_steps 6000 --tokenizer_name bert-large-uncased --per_device_train_batch_size 32 --cache_dir ${OUTDIR}/squad_cache"

echo "===== INF RUN COMMAND: $COMMAND"
eval $COMMAND

echo "===== INF PERF"
COMMAND="python /opt/sambaflow/apps/nlp/transformers_on_rdu/transformers_hook.py  measure-performance -n 10 --inference --model_name_or_path ${OUTDIR}/hf_output_squad_run --tokenizer_name bert-large-uncased --module_name squad --task_name squad --data_dir /usr/local/share/data/bert/SQUAD_DIR/ -p ${OUTDIR}/transformers_hook_inf/transformers_hook_inf.pef --output_dir ${OUTDIR}/hf_output_squad_inference --n-chips 1 --per_device_train_batch_size 1 --per_device_eval_batch_size 32 -b 32 --max_seq_length 384 --overwrite_output_dir"


echo "===== INF PERF COMMAND: $COMMAND"
eval $COMMAND


echo "Machine state after: "
/opt/sambaflow/bin/snfadm -l inventory

echo "Duration: " $SECONDS
