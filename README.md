# Programming New AI Accelerators for Scientific Computing

+ **Tutorial at Supercomputing 2022**
+ **Time:** Monday, 14 November 2022 1:30pm - 5pm CST
+ **Location:** D161
+ **Description:** Scientific applications are increasingly adopting Artificial Intelligence (AI) techniques to advance science. There are specialized hardware accelerators designed and built to efficiently run AI applications. With diverse hardware and software stacks of these systems, it is challenging to comprehend their capabilities, programming approaches, and measure performance. In this tutorial, we will present an overview of novel AI accelerators namely, SambaNova, Cerebras, Graphcore, Groq, and Habana. This includes presentations on hardware and software features on each system. We present steps on how to program these systems by porting deep learning models with any refactoring of codes implemented in standard DL framework implementations, compiling and running on the accelerator hardware. Next, we conduct a hands-on session on SambaNova and Cerebras systems at ALCF AI Testbed. The tutorial will provide the attendees with an understanding of the key capabilities of emerging AI accelerators and their performance implications for scientific applications.

## Agenda

| Time(CST)   | Topic                                                 |
|-------------|-------------------------------------------------------|
| 1.30 - 1.40 | Welcome and Overview of AI Testbed at ALCF, Murali Emani (ANL)      |
| 1.40 - 3.00 | AI Accelerator Vendor Presentations                   |
| 1.40 - 2.00 | Jiunn-Yeu Chen (Intel Habana)     |
| 2.00 - 2.20 | Petro Junior Milan (SambaNova)        |
| 2.20 - 2.40 | Cindy Orozco Bohorquez (Cerebras)      |
| 2.40 - 3.00 | Victoria Godsoe, Ramakrishnan Sivakumar (Groq)        |
| 3.00 - 3.30 | Coffee Break                                          |
| 3.30 - 3.50 | Alex Tsyplikhin (Graphcore)                           |
| 3.50 - 5.00 | Hands on Session with Cerebras and Sambanova          |

## Presentation Slides
The presentation slides and demos are available [here](https://anl.box.com/s/0wltiw52s9yuf0d3gzhjlhgyi841yua1). 


## BERT-Large on AI Accelerators
Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google. Here are instructions to running it on the following system at ALCF AI Testbed:

+ [Cerebras](./cerebras/readme.md)
+ [Sambanova](./sambanova/readme.md)

<!-- To-do: add instructions to use precompiled models and screenshots of the progress and output files -->

## Important Links 

+ [SC22 Tutorial Webpage](https://sc22.supercomputing.org/presentation/?id=tut151&sess=sess221)
+ [Github Tutorial Repository](https://github.com/argonne-lcf/AIaccelerators-SC22-tutorial/)
+ [Overview of AI Testbeds at ALCF](https://www.alcf.anl.gov/alcf-ai-testbed)
+ [ALCF AI Testbed Documentation](https://www.alcf.anl.gov/support/ai-testbed-userdocs/)
+ [Director???s Discretionary Allocation Program](https://www.alcf.anl.gov/science/directors-discretionary-allocation-program)
+ [Join Slack Workspace](https://join.slack.com/t/aiacc-sc22-tut/shared_invite/zt-1i6r49ks1-9IxbIk6NM4TdHaEol26Z9Q)
+ [Paper at PMBS22 "A Comprehensive Evaluation of Novel AI Accelerators for Deep Learning Workloads"](https://sc22.supercomputing.org/presentation/?id=ws_pmbsf120&sess=sess453)
+ [Feedback Form](https://forms.office.com/g/fwaK2kgCt4)

## Instructions for requesting accounts on AI Testbed
(1) Request an ALCF Computer User Account <https://accounts.alcf.anl.gov/accountRequest> if you do not currently have one
(2) If you have an ALCF Account that is currently inactive, submit an account reactivation <https://accounts.alcf.anl.gov/accountReactivate> request*.
(3) If you have an active ALCF account, click Join Project <https://accounts.alcf.anl.gov/joinProject> to submit a membership request*.

Specify the following in your request:
   *  Project Name: aitestbed_training




