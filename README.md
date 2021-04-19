# Evaluation of Deep Learning Based Predictive Models Using Power Systems Use Case

Many important application datastreams can be modeled as a complex graph of entities, where each graph node or entity is associated with a multivariate time-series. The overall system’s behavior is modeled as a dynamical system, and it may evolve through structural changes in the graph, and/or changes in the node-level measurements. Such networks include the power grid, social networks etc. 

This project aims to develop a framework where a distributed network of autonomous agents (subsequently referred to as “holons”) will coordinate to predict states in dynamic networks and which will in turn provide any appropriate alerts/controls to prevent the occurrence of widespread failures. Before the development of the distributed solution, we have first developed and tested two data driven predictive models namely, Koopman Operator Theoretic (KOT)-based model and Graph Neural Network (GNN)-based model using power systems use case with the architecture for the predictive model being centralized. The KOT-based approach (deepDMD) captures the power system evolution as a linear dynamical system on an abstract space. The GNNs model the spatio-temporal correlations using graph convolutional network and are called Spatio-Temporal Graph Convolutional Network (STGCN). The performance of these models in the centralized setting will serve as baseline, and the comparison of performances of the above-mentioned models will help in selecting the best performing one for using in the distributed solution development. 

Both Deep DMD and STGCN  are  trained,  tested  and  compared  rigorously based  on  their  predictions  of  frequencies  in  the  IEEE  68  bus system when subjected to a disturbance. GridSTAGE framework developed at the Pacific Northwest National Laboratory is leveraged to generate multiple datasets (in the form of PMU measurements)for  training  and  testing  by  strategically  creating  load  changes across   the   spatial   locations   of   the   network.   

# For more details about this work, find the Overview.pptx and we refer the readers to the work: ....

# Project Team: Sutanay Choudhury (PI), Sai Pushpak Nandanoori, Soumya Kundu, Seemita Pal, Khushbu Agarwal (PNNL), Yinghui Wu, Sheng Guan (Case Western Reserve University)

# Setup

This repository encloses code for the two models **deepDMD** and **STGCN**. Both models are build on different version of supporting libraries and require separate environment. Please follow below instrucitons to setup environment for respective models.

#### Note : Please first clone the repository on your system.

### Instructions to setup deepDMD model environment

1- Open terminal and run below commands step by step :
	 
	-- conda env create -f Env_Setup_Files/deepDMD_environment.yml  
	-- conda activate deepDMD_env 

2- Run below commands to open Jupyter Lab for either training or testing of deepDMD model:

 	- For training deepDMD model run below command: 
		
		-- jupyter lab Models_Code/deepDMD/deepDMD_Training.ipynb

 	- For testing deepDMD model run below command : 
		
		-- jupyter lab Models_Code/deepDMD/SM_deepDMD_Testing.ipynb

### Instructions to setup STGCN model environment

1- Open terminal and run below commands step by step :
	 
	-- conda env create -f Env_Setup_Files/stgcn_env.yml  
	-- conda activate stgcn_env

2- Run below commands to train STGCN model:
			
	-- cd Models_Code/STGCN_PNNL/
	-- python main.py 
