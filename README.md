# Programming New AI Accelerators for Scientific Computing

+ **Tutorial at Supercomputing 2022**
+ **Time:** Monday, 14 November 2022 1:30pm - 5pm CST
+ **Location:** D161
+ **Description:** Scientific applications are increasingly adopting Artificial Intelligence (AI) techniques to advance science. There are specialized hardware accelerators designed and built to efficiently run AI applications. With diverse hardware and software stacks of these systems, it is challenging to comprehend their capabilities, programming approaches, and measure performance. In this tutorial, we will present an overview of novel AI accelerators namely, SambaNova, Cerebras, Graphcore, Groq, and Habana. This includes presentations on hardware and software features on each system. We present steps on how to program these systems by porting deep learning models with any refactoring of codes implemented in standard DL framework implementations, compiling and running on the accelerator hardware. Next, we conduct a hands-on session on SambaNova and Cerebras systems at ALCF AI Testbed. The tutorial will provide the attendees with an understanding of the key capabilities of emerging AI accelerators and their performance implications for scientific applications.

## Tentative Agenda 

| Time        | Topic                                                 |
|-------------|-------------------------------------------------------|
| 1.30 - 1.40 | Welcome and Overview of AI Testbed at ALCF (ANL)      |
| 1.40 - 3.00 | AI Accelerator Vendor Presentations                   |
| 1.40 - 2.00 | Jiunn-Yeu Chen (Intel Habana)     |
| 2.00 - 2.20 | Petro Junior Milan (SambaNova)        |
| 2.20 - 2.40 | Cindy Orozco Bohorquez (Cerebras)      |
| 2.40 - 3.00 | Victoria Godsoe, Ramakrishnan Sivakumar (Groq)        |
| 3.00 - 3.30 | Coffee Break                                          |
| 3.30 - 3.50 | Alex Tsyplikhin (Graphcore)                           |
| 3.50 - 5.00 | Hands on Session with Cerebras and Sambanova          |


## BERT-Large on AI Accelerators
Bidirectional Encoder Representations from Transformers (BERT) is a transformer-based machine learning technique for natural language processing (NLP) pre-training developed by Google. Here are instructions to running it on AI testbeds - 

+ [Cerebras](./cerebras/readme.md)
+ [Sambanova](./sambanova/readme.md)


## Important Links 

+ [SC22 Webpage](https://sc22.supercomputing.org/presentation/?id=tut151&sess=sess221)
+ [Tutorial Website](https://wordpress.cels.anl.gov/alcf-aitestbed-tutorial-sc22/)
+ [Tutorial Repository](https://github.com/argonne-lcf/AIaccelerators-SC22-tutorial/)
+ [Overviw of AI Testbeds at ALCF](https://www.alcf.anl.gov/alcf-ai-testbed)
+ [ALCF AI Testbed Documentation](https://www.alcf.anl.gov/support/ai-testbed-userdocs/)
+ [Join Slack Workspace](https://join.slack.com/t/aiacc-sc22-tut/shared_invite/zt-1i6r49ks1-9IxbIk6NM4TdHaEol26Z9Q)


<!-- ## Login Information 

+ How to login to CS-2 and SN systems 
  + [Login Cerebras](./cerebras/cs-login.md)
  + [Login Samabnova](./sambanova/sn-login.md)
+ Director’s Discretionary Allocation Program
  + [DD Allocation](https://www.alcf.anl.gov/science/directors-discretionary-allocation-program)

## Experiments 

+ Steps to submit a job and how to see the output
  + [Cerebras](./cerebras/cs-job-submission.md)
  + [Sambanova](./sambanova/sn-job-submission.md)
+ Steps to train BERT-Large model
  + [BERT on Cerebras](./cerebras/cs-bert.md)
  + [BERT on Sambanova](./sambanova/sn-bert.md) -->


<!-- 
#ToDo (Sid)

make readme files for cerebras and sambanova 
add BERT code for cerebras and sambanova 
add particular instructions to run BERT code 
update scripts if necessary for SLURM 
  pipelining vs weight streaming model 
how much time it takes to compile and run the code on each sysyem? 
  how can we reduce waiting time for users during tutorial?  -->


