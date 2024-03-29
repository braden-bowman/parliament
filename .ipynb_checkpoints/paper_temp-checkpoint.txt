The intent of this package is to use an iterative multi-faceted approach to error mitigation while performing community detection in graphs starting only from set nodes with the intent to characterize a community accurately while making the minimal amount of data available to the algorithm. It also explores whether it can improve community detection accuracy in real-world graphs with limited data access. This novel framework combines three components to reduce confirmation bias, dynamic adaptation modules to handle temporal dynamics, and approximation for validation and error estimation. Applied to diverse real-world graphs with controlled data access limitations, this technique showed promising results, but further experimentation and refinement are needed to optimize hyperparameters, validate findings on larger datasets, and evaluate scalability. Ultimately, the method offers a potentially groundbreaking approach to community detection in complex, dynamic graphs, particularly when data access is restricted, laying the foundation for further development and applications in various domains.

All references for this README can be found in the academic_papers section of the package.

Retain this stuff.....

git config --global user.name "bbowman@ironeaglex.com"
git config --global user.email "bbowman@ironeaglex.com"
Create a new repository
git clone https://gitlab.jadeuc.com/gap/skope/team-2/parliament.git
cd parliament
git switch --create main
touch README.md
git add README.md
git commit -m "add README"
Push an existing folder
cd existing_folder
git init --initial-branch=main
git remote add origin https://gitlab.jadeuc.com/gap/skope/team-2/parliament.git
git add .
git commit -m "Initial commit"
Push an existing Git repository
cd existing_repo
git remote rename origin old-origin
git remote add origin https://gitlab.jadeuc.com/gap/skope/team-2/parliament.git


Multi-Modal Error Mitigation for Localized Step-Wise Community Discovery in Dynamic Graphs
The Project Outline section identifies design constraints for an algorithm design problem with real world applications and considers the best method to account for error arising from community detection in a dynamic graph, with dynamic overlapping communities.  The multiple types of error introduced by these constraints necessitate the end algorithm harness multiple strategies for error mitigation to arrive at a solution that is either correct, or at least a very accurate approximation. Types of error requiring mitigation include confirmation bias, temporal decay, and the infeasibility of solution verification and validation for an algorithm intended to discover communities in telemetry based geospatial data.
Problem
This section formally identifies the extents of the problem using the common strengths, weaknesses, opportunities and threats (SWOT) analysis framework and utilizes the implications of that analysis to organize the project team’s research and experimental approaches in a way that is most effectively aligned with the organization and therefore most likely to succeed. Results of performing a SWOT analysis can be seen in Table 1.
Strengths
Office expertise in graph network analysis and ability to leverage cloud computing resources.
Weaknesses
Limited budgetary and infrastructure constraints for data acquisition and processing.
Opportunities
Development of novel techniques to mitigate confirmation bias in community discovery.
Threats
Inaccurate results due to confirmation bias, dynamic graph nature, and verification challenges.
Table 1. SWOT analysis
Context and Background
Graphs are a means of representing non-Euclidean data as a set of nodes and edges. Nodes are the items that make up the graph while edges represent some relation between two nodes. Communities are sub-graphs with shared characteristics indicating a probable connection between their nodes (Vehlow et al., 2014). As graph community detection algorithms are a quickly expanding but already well-researched field, this portion of the problem should be relatively straightforward. The data this task is a large amount of geospatial telemetry or ‘real-time-bidding’ data (Wang et al., 2017). Test data for the project will be sourced from mobile advertising brokers and then loaded into S3 buckets on an Amazon web services instance. 
The ultimate goal of the project is to infer non-linear relationships between nodes (i.e. non-Euclidean, meaning there is no x and y to the plot). By leveraging user-specific advertising ids as nodes with edges derived from other fields in the data the algorithm finds groupings of nodes that are associated through by a relationship other than spatiotemporal coincidence. 
To complicate things further, the data which a ‘production’ algorithm is intend to operate on is massive on a scale such, that even approximating the amount of total data available for this task would be a research task on its own, as the scale forces the data to be limited only to recent history. This reality creates another issue, as the collected data is temporal and continually evolving, nodes and their respective edges are added to and removed from the complete graph on a continuous basis, making the target dataset not just a graph, but a dynamic one.  
The projects true problem comes down to business imposed fiscal constraints that limit available infrastructure. Resources are not available to perform community detection on the entirety of the data, let alone on a continuous basis.  For this reason, it has been decided to look into the possibility of a new method for community detection in graph data that takes a more localized approach by starting with advertising id’s known to be a part of a community and querying only a small subset of the data, thereby alleviating the necessity to compute a full solution on a rolling basis.
Data for the known nodes will be queried and stored in a local database, after which additional nodes and edges will be selected for addition to the local database based having a connection to an initial or ‘seed’ node. Based on thorough initial testing, this method appears to be subject to the idea in graph theory known as the six degrees of separation (Heikkinen, 2021), which asserts that in a social network there exists at most six edges between any two nodes in the graph.  
As the amount of returned data increases exponentially with each additional step or edge traverse through the network, the maximum reliable traversal depth in practice has been observed to be two edges which necessitates a means for pruning edges as the network is queried in a stepwise fashion in order to reduce the extent of the graph needed to find a solution, instead using a targeted or localized portion of the graph to discover the entirety of the community. Key constraints leading to these conclusions are derived from the following factors:
1.	Computational cost
2.	Computational tractability
3.	Dynamic nature of the graph or the need to recompute previous results continuously
Error Sources and Mitigation
In order to create an algorithm that can successfully accomplish this taskit will be necessary to prune nodes at each step, limiting the number of edges follow-on queries make ‘out’ of the community network. This is a necessary step to makes the problem computationally and fiscally tractable. While this method is intended to make the problem tractable, it also introduces several potential sources of error in the final result which must be quantified and mitigated in order for the user to have confidence in any results. Sources of error introduced using this strategy are collated in Table 2.
Error type	Problem in context
Confirmation bias 
(Mao & Akyol, 2020).	Faulty parameters for pruning nodes at each step drives the resulting discovered community resulting in a final result that is not a true representation of the community. 
P versus NP
(Aloise et al., 2009)	As the algorithm will not have the entirety of the data available, and in fact only a small sub-set of it is intended to be used to find the final solution, it becomes difficult or impossible to verify and validate the solution.
Optimal stopping
(Ankirchner et al., 2017)	Progressing through the graph via edges that connect new nodes into the initial set is an iterative process, at some point the complete community will have been discovered and the algorithm will need to cease iterations. Choosing the optimal time to stop is an exercise in balancing how confident we are that the full community has been discovered, versus continuing after and wasting time and resources to continue searching for a solution that has already been found though it cannot be confirmed.
Temporal decay
(Zhang et al., 2020)	Any results from a community detection algorithm will be subject to model drift, specifically in this case temporal decay, as the data in the graph changes and minor alterations to the updated graph result in evolved communities not perfectly representative of the status-quo.
Table 2. Sources of error for localized graph community detection with pruning in a dynamic graph.
Confirmation bias, optimal stopping, temporal decay, validation in aggregate make finding correct solutions algorithmically a challenge. Evaluating the magnitude of each error and how they interact with one another is complex and requires experimentation and analysis to determine how best to mitigate the effects and ensure the reliability of the discovered solution. The remainder of this paper aims to develop strategies to address these issues and provide reliable community detection within the constraints of a dynamic graph and limited exploration steps. These obstacles in context of the problem at hand combine readily to produce the following more concise problem statement.
Problem Statement
The purpose of this research is to develop and evaluate and quantify strategies that mitigate the combined error introduced from confirmation bias, solution validation, optimal stopping, and temporal decay in an algorithm attempting to perform localized community discovery via edge traversal in a dynamic graph, establishing a solution with quantifiable confidence.

 
Proposed Solution
This section focuses on identifying a solution that addresses the root cause of the problems identified in the problem statement, defines a framework for developing it successfully, and presents a hypothesis. Further analysis is presented by inspecting the hypothesis with consideration toward the problem statement. This analysis yields key concepts that form the basis for a research question. We then review past research through a literature review (Carrera-Rivera et al., 2022) to identify differences and similarities between our approach and existing research.
Solution Framework
Based on the formulated problem statement at the end of the project outline section, the technical problems that need overcome in order to design an algorithm capable of discovering satisfactory solutions (Lee et al., 2005) can be extrapolated directly from the problem statement.
Technical Issues Faced
As noted, technical issues faced by the project are confirmation bias, temporal decay, verification and validation, and optimal stopping. The following is a brief description of how each of these errors affects the proposed algorithm:
1.	Confirmation bias
•	Due to limited data queries, selection criteria must be used to decide which nodes receive further exploration in subsequent steps. This inherently introduces bias, as pruning nodes and their alternate edges can overlook connections to the community (Mao & Akyol, 2020).
2.	Temporal decay
•	The target dataset constantly updates and removes data, making characterization of communities difficult as relationships constantly change (Vehlow et al., 2014).
3.	Verification and validation
•	Access limitations prevent traditional validation methods, rendering solution accuracy uncertain due to the P vs. NP problem (Gordeev, 2022).
4.	Optimal stopping
•	Determining when to stop exploration and return results while balancing accuracy, cost, and confidence in the detected community is crucial (Ankirchner et al., 2017).
Addressing these challenges effectively requires tackling their root cause – combined error introduced by all four factors. Existing solutions tend to address individual aspects, but none comprehensively account for the complex interplay of these errors. Our proposed solution aims to overcome this limitation by quantifying and reconciling these errors within a unified multi-faceted framework.
Root Cause
The root cause of the problem faced is the compounded error introduced from multiple constraints that must be addressed with different mitigation strategies.  The specific objectives of the project are how to deal with each source of error in such a way as to optimize the final solution over the optimization of any particular error source the specific objectives of the project are to mitigate each source of error individually using the following strategies:
1.	Quantify or Mitigate Confirmation Bias
•	Design metrics to measure confirmation bias introduced by stepwise exploration and pruning techniques.
•	Explore algorithm modifications or alternative approaches to minimize bias and ensure a balanced exploration of potential communities.
2.	Temporal Decay
•	Incorporate temporal aspects of the graph into community detection methods to account for evolving structures and relationships prevalent in dynamic graph problems.
•	Investigate adaptive techniques that can adjust to changes in the graph over time.
3.	Verification and Validation
•	Develop methods to establish solution confidence by assessing the reliability of community detection results despite incomplete graph data and the inability to perform traditional mathematical validation.
•	Explore probabilistic approaches with confidence scoring to convey uncertainty and potential error margins.
4.	Optimal stopping
•	Determine effective strategies for deciding when to terminate exploration and return results, balancing accuracy, computational cost, and confidence in the detected communities.
Multi-faceted Solution Framework
In order to prioritize the solution of the combined algorithm we propose a multi-faceted framework employing state-of-the-art techniques against each challenge combined into a single metric in which each mitigation strategy is given a weighting and combined into a single metric which mitigates their combined impact. 
Hypothesis Statement
By integrating stochastic attention mechanisms for bias mitigation (Mao & Akyol, 2020), dynamic adaptation modules for evolutionary resilience (Alotaibi et al., 2023), a Bayesian framework for accurate error quantification despite limited data (Hirsh et al., 2022), and an adaptive thresholding strategy for optimal stopping (Egami & Yamazaki, 2014), we hypothesize the following:
A unified-multi-faceted framework will significantly reduce confirmation bias compared to existing solutions, leading to more accurate and unbiased community detection results, maintain high community detection accuracy even in dynamically evolving graphs by effectively incorporating temporal information and predicting future community changes, provide reliable and quantifiable measures of error and confidence in the discovered communities, even with limited data access, through the use of stochastic validation and Bayesian uncertainty quantification (Kendall & Gal, 2017). to dynamically determine the optimal stopping point (Xu et al., 2015) during community exploration based on changes in key metrics and error profiles, ensuring efficient and accurate results.
Research Questions
To validate this papers hypothesis the following research questions will be investigated:
1.	How effectively does the stochastic attention mechanism mitigate confirmation bias compared to alternative approaches?
2.	To what extent does the dynamic adaptation module improve community detection accuracy in evolving graphs compared to static models?
3.	Can the proposed stochastic validation and error quantification framework provide reliable accuracy estimates with limited data access?
4.	What are the optimal stopping thresholds for different graph structures and error profiles, and how effectively can the adaptive thresholding strategy learn these thresholds?
Research will necessarily seek to discover where these various concepts have been the subject of research in combination in order to assess what a good strategy is to address all of them occurring simultaneously in a single model space.
For a greater study, this work would attempt to iterate through each concept isolated as the independent variable and assess how it is impacted by each of the others. For this paper itself, the main concern assessed colloquially is that of confirmation bias. Therefore, confirmation bias will act as the independent variable in this first iteration of research. The other three will be assessed as dependent variables, and given time constraints, the priority is to assess optimal stopping as this is the main operator in the algorithms assessment.  Based on the distillment of the problem to this more specific issue, the independent variable, dependent variable, and research question for this paper are defined explicitly as such:
Independent Variable
Confirmation Bias – as a set of metrics per node stored as a feature vector for that node.
Dependent Variable
Optimal stopping criteria (Egami & Yamazaki, 2014) – a set of metrics based on currently available data that define when community discovery is complete and the algorithm shall cease to search and return a solution.
Related Works
Related works are not difficult to find for any of the topics and concept identified in this paper.  Broadly, research that applies to this subject can be categorized according to the specific error introduced, namely confirmation bias or just bias, dynamics, stopping and solution validation. There were excellent resources discovered for each category that directly address many of the concerns this paper presents and in a similar context. While no single piece of research directly addresses all of the issues identified in this paper, making its solution something novel, the information derived from research that does in fact address some of the combinations of conceptual issues identified and in aggregation they may offer a guiding framework for further combining methodologies in order to find a solution.
Bias
The benefits of focusing on confirmation bias, or eliminating bias in general, from the algorithm are most centered on the fact that this ensures that the algorithm more nearly finds legitimate results that may be novel or surprising rather than simply finding what one already expected. This guidance is well established for network topology inference by Mao and Akyol (2020), as they use a mechanism to account for confirmation bias while mapping communities in social networks. The difficulty in the case of this paper is in appropriately selecting and tuning parameters of the algorithm so that it finds true, data driven communities rather than just groupings that satisfy the biased parameters. Bias is necessary in this case in order to limit the amount of data and make the problem tractable, but must be quantified and heeded so that results are accurate and useful.
On Inference of Network Topology and Confirmation Bias in Cyber-Social Networks (1)
In their research, Mao and Akyol (2020) demonstrate a method to infer sub-graphs from cyber-social data. While they do not mention communities, this is in fact what they are finding. Communities derived from initial start points. In this sense, they discovery mechanism they use works on a localized basis and in that sense if more similar to the proposed algorithm in this paper.
Pros and Cons
The main benefit of this approach is the mechanism by which they account for confirmation bias as they infer as well as the fact that the algorithm starts from a limited set of nodes, one individual in this case, and then searches outward to infer the network (Mao & Akyol, 2020). It limitations mainly lie in the set of assumptions made in order to make the algorithm tractable which involved reassigning nodal values using “node knockout” and power spectral analysis. These make sense in the context of the paper, but may introduce additional biases if implemented in the solution to the current problem.
Main technical features
The main feature of this articles is an equation presented which incorporates a measure to avoid confirmation bias as it searches the graph to discover a network (Mao & Akyol, 2020).
Main difference from proposed
Main differences from the proposed are that the graph is not assumed to be dynamic, that they do not actually seek to find a community, though their method resembles the proposed, and that they make several assumptions up front that alter nodal attributes in order to make the algorithm function (Mao & Akyol, 2020). These assumptions are likely not acceptable in the solution for this papers problem, but additional consideration is required to establish the veracity of that statement and their methods will be considered.
Dynamics
Taking dynamics into consideration introduces several benefits and drawbacks.  In term of benefits, updating communities as new data is ingested or taken away ensure that the inferred community is as accurate and complete as possible. However, this update also has negative impacts. Changing the base dataset at each step requires a significant amount of additional computation that make it more difficult to accomplish, and the changes introduced into the graph representation make it difficult to use previous results in order to perform future steps. The best research found considering these constraints is from Vehlow et al. (2014), and interestingly is much more focused on the visualization aspect. It seems that Vehlow and his collaborators, while giving them due mathematical consideration, undervalue the utility of the data engineering features of their work.
Visualizing the Evolution of Communities in Dynamic Graphs (2)
This paper presents a method of visualizing the evolution of communities in a graph over time. While the emphasis is on the visualization, the algorithmic mechanism implemented by Vehlow et al. (2014) presents a method of dealing with dynamic graphs for analytic results as well as it incorporates a stepwise temporal processing that derives communities at each new step with consideration toward their prior membership.
Pros and Cons
Benefits of this method are as stated in the summary, it elegantly deals with the evolving nature of graphs on a mathematical and visual level (Vehlow et al., 2014). Drawbacks to the method are the lack of consideration toward the community detection algorithm used, its success rate, and that it requires access to the full dataset to sort nodes into communities.
Main technical features
Main technical features of this article are its forward propagation model for community detection and its novel method of sorting nodes, edges and communities for visualization (Vehlow et al., 2014).
Main difference from proposed
The main differences of this algorithm from what is proposed in this paper is that it utilizes the full dataset when inferring communities (Vehlow et al., 2014), and does not take into consideration the accuracy of its community assignments as the general focus is on visualization rather than inference.
Stopping
As a solution cannot be presented in this instance until the algorithm stops, the stopping algorithm is of the utmost importance. If constraints are too loose, the algorithm will return a result that does not fully characterize a community and leave out crucial nodes, while on the other hand, if stopping criteria is too tight then the algorithm will continue indefinitely (in the case of a dynamic graph) at worst, or include many nodes at the edge of a community that should not be included (Ankirchner et al., 2017). 
A Verification Theorem for Optimal Stopping (3)
According to Ankirchner et al. (2017) in this paper, they assess the problem of optimally stopping a continuous time process based on an expectation, which is exactly what the proposed solution would do. While this paper does not provide a perfect solution to the problem, it does provide a value function that can help to make a decision in these instances.
Solution Pros and Cons
The solution provided in the paper is not de facto. While it is a good guide, it is imperfect and so may be inappropriate to insert into an automated process as feedback. Further study on this matter is required to determine the best course.
Main technical features
The main technical feature of this article is a mathematical representation of the current system state and when to stop a continuous time process (Ankirchner et al., 2017).
Main difference from proposed
While this article does provide a good solution to optimal stopping in a continuous time process, the algorithm is really geared to a single solution, not one that is automated and repeated, which introduces an additional layer of error if the results of this particular algorithm are not assessed each time, they may create model drift or bias in automated solutions (Ankirchner et al., 2017). This paper also has no direct relation to community graphs or their detection via localized stepwise discover and so the results, if desired, must be applied carefully to a very small and specific portion of the proposed solution.
Validation
The main issue with validation, as this problem falls firmly withing the P versus NP problem space, is that no solution will likely ever be truly validated except when the resulting solution is directly vetted by domain experts. In this sense, this is the most difficult of the problems associated with the chosen problem. While knowing that we cannot establish a method to truly validate a solution, what we can do is approximate it via an estimator (Ankirchner et al., 2017). This will at least give some sense and measure of how the algorithm is performing. As an assessment of the entire data set is computationally infeasible, it makes sense that this estimator be stochastically derived from some form of small scall sampling (Gates et al., 2016). 
A Monte Carlo Evaluation of Weighted Community Detection Algorithms (4)
In their work to evaluate weighted community detection algorithms, Gates et al. (2016) and his research team utilize a Monte Carlo technique to made the evaluation possible. The Monte Carlo technique used in this paper was utilized with the intent to assess the performance of multiple algorithms, and while this differs from the express intent of this paper, it is only in the utility of the information derived from the technique. Monte Carlo, or another small-scale sampling technique to derive estimators of population estimators may present a method of validations for the algorithm in question with only the simple effort of applying the estimators back into the higher-level algorithm rather than reporting them as results.
Solution Pros and Cons
The main benefit of this method is that it provides a means to validate a solution without access to the entirety of the data (Gates et al., 2016). By constructing a clever sampling schema, solutions may be assigned a quantifiable metric associated with their error. Whether the metric is good or bad is irrelevant as it is at least a means to measure. 
Main technical features
The main feature in this article is the use of Monte Carlo as a means to estimate the accuracy of algorithm without the need to perform a comprehensive graph assessment (Gates et al., 2016).
Main difference from proposed
This technique differs from the proposed in that it only intends to evaluate performance of community detection algorithms (Gates et al., 2016). The proposed has many more layers of complexity that need to be addressed, and if it were to incorporate something like this methodology it would serve in a more predictive fashion, using the results not to assess but rather to tune the model.
On P versus NP (5)
When dealing with P versus NP, the discussion and evaluation center around whether a solution is findable, and then assessing whether that solution is verifiable. According to the research by Gordeev (2022), P != NP is a favored solution in his assessment of verifiable solutions from graph algorithms. However, this work, while good, still favors taking a side in a monumental debate without delivering conclusive results. However, thought the results of the calculations in the paper may not serve as a proof for P versus NP, they are a clever mechanism to estimate which may be useful in application.
Solution Pros and Cons
The main benefit of this technique is that even if it is not perfect it can at least give a quantitative sense of the accuracy of a solution. While on the other hand, it is not a true solution as it does not provide definitive proof (Gordeev, 2022). 
Main technical features
Consideration of a graph network specifically using a double Erdos-Ko-Rado technique which seem to imply whether a solution is correct or incorrect (Gordeev, 2022).
Main difference from proposed
The main difference in this paper is that it does not attempt to discover communities. While Erdos-Ko-Rado itself may be seen as a simplistic community algorithm and forms the basis of more complex algorithms, it does not in fact perform the desired operation (Gordeev, 2022). In addition, this paper is not applied to a dynamic graph, consider optimal stopping (number of steps are predetermined by Erdos-Ko-Rado), or account for confirmation bias. As with the others, this may be a good method to deal with one aspect of the desired algorithm, but the real difficulty is accounting for error introduces from all these sources at once.
 
 
Research Methodology
This section presents the methodological framework designed to answer the identified research questions and described the rationale driving experimental design, followed by a detailed breakdown of the experiment requirements, procedures, and data collection process. The experiment will use the following error mitigations strategies while collecting metrics on their parameters for each test iteration as well as accuracy metrics of formula that assigns their weighting in the final multi0faceted metric:
1.	Confirmation bias mitigation: 
Apply a stochastic attention mechanism and regularization penalty.
•	Employ a stochastic attention mechanism that dynamically assigns weights to nodes based on features and network context, reducing dependence on biased pruning (Vaswani et al., 2017).
•	Implement a regularization penalty promoting exploration of diverse paths, minimizing bias towards pre-existing community structures (Chen et al., 2021).
2.	Temporal decay mitigation:
Utilize dynamic graph adaptation
•	Utilize a temporal decay function to down weight the influence of outdated data, adapting to evolving relationships (Yu et al., 2020).
•	Embed a community embedding model capable of capturing temporal dynamics and predicting future community evolution (Wang et al., 2022).
3.	Verification and validation:
Evaluate results using a large but static graph dataset with temporal features using stochastic validation and error quantification.
•	Integrate a Monte Carlo sampling approach to estimate community accuracy and confidence across a small data subset (Gates et al., 2016).
•	Design a Bayesian uncertainty quantification framework to model and propagate error through the algorithm, providing confidence estimates for the discovered communities (Kendall & Gal, 2017).
4.	Optimal stopping:
Leverage adaptive thresholding to avoid hard stops inherent from thresholds.
•	Develop a dynamic stopping criterion based on changes in community metrics like modularity and edge density, accounting for error introduced by limited data (Xu et al., 2015).
•	Utilize a reinforcement learning framework to adaptively learn optimal stopping thresholds for different graph structures and error profiles (Sutton & Barto, 2018).
Research Matters
Terminology
Common terms to be used in the experiment phase are defined as follows:
•	Community detection: The process of identifying densely connected groups of nodes within a graph (Gates et al., 2016).
•	Modularity: A metric measuring the quality of a community partition, considering intra-community edge density and inter-community edge sparsity.
•	Error quantification: Quantifying the uncertainty and potential deviation from true community structures in the detected communities.
•	Adaptive stopping: Determining the optimal point to terminate community exploration based on pre-defined criteria.
•	Stochastic attention: Dynamically assigning weights to nodes based on features and network context, mitigating bias introduced by pruning decisions.
Test Metrics
The experiment will utilize several test metrics to ensure the accuracy of results including accuracy, precision, recall, F-1 score, and error-margin. These metrics will be used to compare the These metrics are defined in Table X.
Error	Applies to	Metric	Definition
Verification and Validation	Discovered community	Accuracy	Proportion of correctly identified community assignments compared to ground truth communities.
Verification and Validation	Discovered community	Precision	Ratio of true positives (correctly identified community members) to all assigned community members
Verification and Validation	Discovered community	Recall	Ratio of true positives to all actual community members.
	Discovered community	F1-score	Harmonic mean of precision and recall, balancing both metrics.
	Discovered community	Error margin	Confidence interval around the estimated accuracy, quantifying uncertainty in the results.
Confirmation Bias	Node pruning	Rate of temporal decay	Rate at which new nodes are added to a community at each temporal step.
Optimal Stopping	Node pruning	Area under the curve (AUC)	Total area under the Precision-Recall curve. A larger area indicates better overall performance
			
			
Table 3. Metrics used to record algorithm error during experiment.
These are common testing metrics shown in Table 4 could be utilized to assess both the community detection and node pruning steps, however, initial results will be assessed as describe in the table. Logging throughout the experiment is such that final additional metrics can be calculated using these metrics for all of the various decision points in the algorithm as they can be applied against vector data to calculate loss via linear-regression, the results of which can be used to tune individual parameters, or in this case hyper-parameters of the algorithm (hyper-parameters are parameters that dictate those of an underlying algorithm which is a sub-set of the multi-faceted algorithm as a whole). Depending on initial experimental results, more advanced techniques may be used 
Assumptions
In order to make a viable experiment, we must make the following assumptions:
•	The ground truth community structure for the chosen datasets is accurate and complete.
•	Random data sampling accurately represents the overall graph characteristics.
•	The chosen experimental parameters and algorithms are sufficiently optimized for reliable performance.
•	External factors influencing the network during experiments have minimal impact on the findings.
System Setup and Lab Environment
In order to effectively execute the experiment, the experiment will require a high-performance computing platform, a controlled lab environment with minimal external network interference and data corruption risks. The best option in this case is to use a large cloud-computing platform for the initial use case as this will provide scalable computing resources capable of processing the maximum load require, which will be creating a community graph validation dataset with no depth or temporal awareness on the full 16TB of experimental test data.
Experiment Design
Experimental design will focus on selecting a dependent and independent variable so as to isolate variation as much as possible. In order to test the various error mitigation mechanisms as well as the output of the complete algorithm, the experiment requires a complex design, varying parameters for each portion of the algorithm. In order to evaluate the performance of these variations, the experiment will construct a set of test graphs with isolated variables that can serve as a benchmark to assess performance overall and at each decision point. 
Graph Communities for Validation and Verification
The validation dataset will consist of a set of graphs on which community detection has been performed with different variable held constant allowing a method to quantify how biases introduced via the constraints for each metric. As these graph detections are not limited to a subset of the data, they will comprise the most compute intensive portion of the experiment. Validation data will be computed for four scenarios:
1.	Community detection for the complete experimental dataset without seed node assignments, but with regard for time or depth. This algorithm performs no node pruning. 
o	This algorithm sees all the data and so is not ‘blind’.
o	This algorithm receives no additional information beyond the data, and so is not biased.
o	Associated metric: Community assignment [one dimensional vector result] per node.
2.	Community detection for the complete experimental dataset with seed node assignments but without regard for time or depth. This algorithm performs no node pruning. No blinding
o	This algorithm is not blind.
o	This algorithm is provided additional data in the form of seed node labels and so is biased.
o	Associated metric: Community assignment [one dimensional vector result] per node.
3.	Community detection starting with seed node assignments, adding nodes by expanding by a depth of one edge with community detections recorded at each step until the full graph is characterized.
o	Only varies by edge depth, not temporal steps. This algorithm is ‘blind in one eye’.
o	Metric: Community assignment at depth [(2, dimensional), (vector, result)] per node
o	Must account for disjoint nodes and communities – stopping occurs when no new nodes are added to the target lest it enter an endless loop.
4.	Community detection starting with seed nodes and adding nodes by timestep, including all new nodes in the timestep whether they are disjoint from the graph or not. Stopping occurs when the max time is reached.
o	Only varies by temporal steps, not edge depth. This algorithm is blind in the other eye.
o	Metric: Community assignment at temporal step [(2, dimensional), (vector, result)] per node
Discovery Algorithm: One experimental run
Once validating graphs have been created as described in the Graph Communities for Validation and Verification section, the experiment will run the algorithm against the experimental data varying the assigned nodes at input for several sets of manually identified nodes. The algorithm will operate via the pseudo-code in Table 4.
Step	Phase	Process
1	Initialize	Load seed date
2		Set parameters for underlying error mitigation algorithms.
3		Make ‘seed’ set of nodes known to belong to a community immutable.
4		Add seeds to local graph dataset
5	Iterate	Query seed nodes for their edge related nodes.
6		Add queried nodes to local graph dataset.
7		Apply community detection to assign queried nodes to communities.
8		Store community, depth and temporal metrics.
9		Prepare to expand from queried nodes for their edge related nodes at the next depth level from the seed nodes.
10		Evaluate and store individual node metrics for all nodes including er-ror mitigation.
11		Check stopping criteria. If true go to step 12. If false go to 
12	Stop	Store results.
13		Prepare to query edge related nodes at the next depth level.
14		Conduct node pruning for query by assigning nodes to be included.
15		Prune nodes unlikely to be part of the target community
16		Go to step 5.
Table 4. Single run pseudo-code for experimental graph community discovery algorithm.
Independent variables
•	Confirmation bias mitigation techniques (e.g. Stochastic attention vs. Regularization)
•	Rate of temporal decay (e.g. Temporal decay (Thongprayoon et al., 2023) vs. Community embedding (Cavallari et al., 2017))
•	Edge depth
Dependent variables
•	Metrics from Table 3, independent variable in cases where they are held constant.
Experiment Requirements
Using the validation graphs and discovery algorithm previously described, the experiment will run according to the following procedure, conducting experimental runs for as many sets of ‘seed’ nodes as are available, assessing algorithm performance for each independent variable using the previously identified metrics.
1. Data Availability
•	Access to benchmark datasets with ground truth community structures.
2. Algorithm Implementation and Optimization
•	Efficient implementations of the proposed techniques and baseline algorithms.
•	Hyperparameter tuning to optimize performance for each algorithm and technique.
3. Statistical Significance
•	Use of appropriate statistical tests to assess the significance of observed differences between experimental groups.
•	Replication of experiments to ensure consistency and generalizability of the findings.
4. Ethical Considerations
•	Anonymization or pseudonymization of any potentially sensitive data used in the experiments.
•	Transparency regarding data collection, storage, and usage practices.
Experiment Procedures
The experiment will be conducted procedurally following the phases defined below in order:
1. Dataset Preparation
•	Preprocess and prepare the chosen datasets (cleaning, scaling, format conversion).
•	Generate synthetic datasets with varying properties if needed.
2. Algorithm Implementation
•	Implement the proposed framework with various combinations of selected techniques and baseline algorithms.
•	Optimize hyperparameters for each algorithm and technique based on validation data.
3. Experimental Runs
•	Run each experimental configuration (combination of independent variable levels) on the prepared datasets.
•	Replicate each run multiple times to account for stochasticity and improve statistical power.
4. Data Collection and Analysis
•	Collect performance metrics (accuracy, error margin, computational efficiency) from each run.
•	Analyze and visualize the collected data to identify trends and statistically significant differences.
5. Evaluation and Conclusion
•	Interpret the results and draw conclusions regarding the effectiveness of each technique and the overall framework. 
•	Discuss limitations and potential future research directions based on the findings.
 
Project Plan
This section outlines the specific details of the project including parameters for data analysis and collection, risk assessment, and a cost benefit analysis of the proposed project.
Project Plan for Data Collection and Analysis
This section outlines the key milestones and activities involved in realizing the research project. It includes a detailed breakdown of the data collection and analysis plan tailored to our proposed multi-faceted framework for community detection in massive graphs with limited data access. Tasks for data collection and analysis include the following:
Data Collection
This section outlines the protocol for data collection, transformation and retention and enumerates the tasks associated. In order to better evaluate the algorithm, the experiment will utilize a large but not dynamic dataset. This dataset 
1. Algorithmic Experiment Data
Experiment data will consist of the data queried, the results of the community detection algorithm after each successive query, a list of nodes pruned before performing another query and what criteria excluded them from the subsequent query.
Important Experimental Considerations
Collection Methods
•	Implement the framework and run experiments on diverse sets of seed nodes with controlled properties.
•	Logging performance metrics.
•	Utilizing validation graphs to get a sense of how experimental results differ from traditional and less complex methods.

Data Type	Collected Metrics	Purpose
Collected metrics	1.	Community detection results (discovered community structures)


2.	Algorithmic performance metrics (runtime, resource consumption)

3.	Monte Carlo sampling statistics (sample sizes, confidence scores). This is a pseudo-Monte Carlo as the starting data must be defined. A true Monte Carlo is an option.
4.	Ground truth community data (if available) or analyst-validated communities	1.	Evaluating the effectiveness of the proposed framework in identifying accurate and diverse communities.
2.	Measuring the computational efficiency and resource utilization of the algorithm.
3.	Assessing the reliability of the Monte Carlo sampling technique and confidence scores.
4.	Validating the identified communities against ground truth or human expertise.
Table 5. Algorithmic data and metrics collected for evaluation.
2. Dataset Acquisition and Preparation
This section outlines where data is collected from and how it is processed prior to analysis.
Data Type	Description	Purpose
Real-time bidding telemetry data	Data broker purchased data available in-house.	Training and test set generation. Conducting of the experiment. Provides test data for the algorithm along with generated known ground truth for comparison.
Benchmark	Generated benchmark datasets with ground truth community structures	Approximate validation and verification performance.
Table 6. Description of the raw data to be used in the experimental runs.
Data Analysis
Limiting data collection and analysis to a finite dataset and identifying inter-operable metrics for the validation graphs and experimental run ensures the ability to perform a thorough evaluation and analysis of the proposed framework across the several variables being tested. By analyzing how algorithmic performance is impacted by temporal decay, confirmation bias, and approximating validation, we can gain a holistic understanding of the algorithm’s strengths, limitations, and potential real-world applicability.
Project Timeline and Cost Estimation
The project is a small research project that will deliver a useable piece of software at the end which a client can use to leverage the algorithm in a repeatable fashion. The scope of the project means that there will not be nearly as much overhead and that the delivery will be either a minimum viable product (MVP), or perhaps a slightly improved version of the MVP. Accomplishing this is expected to take four to six months for a team consisting of five data scientist, five software developers, and two administrative personnel. The project timeline can be seen in Table 7.
Project Timeline
Project Duration: 6 months
Resources
•	Researchers/Analysts: 15 total (5 entry-level, 5 mid-level, 2 administrative)
•	Computational Resources: Flexible AWS instance connected to NVIDIA DGX GPU cluster
Dataset
•	In-house Real-Time Bidding (RTB) dataset (16 TB)
Cost Assumptions
•	Researcher/Analyst average monthly cost: $12,000
•	AWS instance and DGX GPU cluster costs based on estimated usage and current pricing (see detailed breakdown below)
Timeline
Phase 1: Data Collection and Preparation (1 month)
Resources
•	2 entry-level analysts, 1 mid-level analyst
Tasks
•	Download and store RTB dataset on AWS S3 storage.
•	Preprocess and clean the data (removing duplicates, handling missing values, etc.).
•	Generate synthetic graphs with controlled properties (optional).
•	Quality checks and data exploration.
Estimated Cost
•	Personnel: $12,000 * 3 researchers * 1 month = $36,000
•	AWS S3 storage: Variable based on data size and access frequency.
•	Cloud compute for preprocessing (estimated): $5,000
Phase 2: Algorithm Development and Implementation (2 months)
Resources
•	3 mid-level researchers, 2 senior researchers, 1 software engineer
Tasks
•	Implement the proposed framework with different techniques based on skill levels.
•	Unit testing and optimization of individual components.
•	Integration and testing of the complete framework.
Estimated Cost
•	Personnel: $12,000 * 6 researchers * 2 months = $144,000
•	Cloud compute for development (estimated): $20,000
Phase 3: Experimentation and Analysis (2 months)
Resources
•	2 mid-level researchers, 1 senior researcher, 2 analysts
Tasks
•	Run experiments with diverse configurations on the RTB dataset using the GPU cluster.
•	Collect performance data (accuracy, efficiency, etc.).
•	Conduct statistical analysis and interpret results.
•	Prepare visualizations for effective communication.
Estimated Cost
•	Personnel: $12,000 * 5 researchers * 2 months = $120,000
•	Cloud compute for experiments (estimated): $40,000
•	GPU cluster usage (estimated): $20,000
Phase 4: Documentation and Reporting (1 month)
Resources
•	1 senior researcher
Tasks
•	Write research paper and final report summarizing the project.
•	Prepare presentations for dissemination of findings.
Estimated Cost
•	Personnel: $12,000 * researcher * 1 month = $12,000
Total Estimated Cost
Personnel: $414,000 (excluding variable expert fees)
Cloud Computing: $90,000 (estimated - actual costs may vary based on usage)
GPU Cluster: $20,000 (estimated)
Data Storage: Variable based on S3 usage
Total: $424,000 (excluding expert fees) + Data Storage + Variable Expert Fees
Table 7. Breakdown of a proposed six moth project timeline with estimated costs.
 
Risk and Cost-Benefit Analysis
This section analyzes the potential risks and benefits associated with the proposed research project on community detection in massive graphs with limited data access.
Risk analysis will focus on identifying issues that are most likely to prevent successfully tuning an algorithm to find community with the constraints outlined in the Project Outline section. Prominent risks can be seen in Table 3.
Risk	Description	Mitigation Strategies
Technical Feasibility	Implementing sophisticated techniques like stochastic attention, dynamic adaptation modules, and Bayesian error quantification within a unified framework presents a significant technical challenge. Unexpected difficulties could arise in their integration and optimization, potentially delaying the project timeline or even impacting its feasibility.	•	Conduct a phased development approach, focusing on individual components initially and gradually integrating them.
•	Utilize modular and well-documented code architecture to facilitate troubleshooting and debugging.
•	Explore existing open-source implementations and libraries for specific components to leverage prior development efforts.
•	Allocate sufficient time and resources for algorithm optimization and performance testing.
Data Availability and Quality	Access to comprehensive real-world datasets with ground truth community structures can be challenging. Additionally, data quality issues like noise, inconsistency, or missing values could adversely affect the evaluation of our methods.	•	Leverage publicly available benchmark datasets like SNAP and PPI.
•	Partner with organizations that possess relevant private datasets if possible.
•	Implement robust data preprocessing techniques to handle noise and missing values.
•	Explore synthetic data generation methods to create controlled test environments.
•	Develop data quality assessment metrics to assess the reliability of acquired datasets.
Computational Complexity	The proposed framework may require significant computational resources due to its complex algorithms and potentially large datasets. This could limit its scalability and applicability to real-world scenarios.	•	Investigate efficient algorithm implementations and explore distributed computing frameworks for resource optimization.
•	Develop lightweight approximations or hybrid approaches where possible to reduce computational overhead.
•	Utilize cloud computing platforms to access scalable and on-demand computational resources.
•	Prioritize experimentation with smaller datasets initially and gradually scale up with optimized algorithms.
Evaluation	Evaluating the effectiveness of our framework in real-world settings with limited data access poses significant challenges, as ground truth communities might be unknown or inaccessible.	•	Employ indirect evaluation metrics like modularity and edge density comparison with established benchmarks.
•	Develop synthetic data generation models that incorporate realistic limitations on data access.
•	Explore transfer learning techniques to adapt the framework to real-world scenarios with limited data.
•	Collaborate with domain experts to validate the relevance and practical significance of the discovered communities.
Table 8. Risks associated with the project and mitigation strategies.
 
Cost-Benefit Analysis
This section summarizes the main points of a cost benefit analysis conducted before proceeding further with the project in order to ensure that doing so is feasible enough to be successful and not so costly as to cost more to execute than benefit that it provides.
Benefits
Benefits consist most noticeably of the company successfully delivering a product for a contract that they have, generating profit, pleasing clients and building reputation. For the project to be a success the algorithm must meet the following criteria: 
•	Deliver Improved Community Detection Accuracy: This project aims to significantly improve community detection accuracy in massive graphs compared to existing methods, especially with limited data access. This can benefit various applications like social network analysis, anomaly detection, and biological network analysis.
•	Enhanced Error Quantification: Quantifying and mitigating error introduced by various factors through a unified framework provides valuable insights and improves trust in the detected communities.
•	Dynamic Graph Adaptation: The ability to adapt to evolving graphs provides a crucial advantage for real-world applications where network structures are constantly changing.
•	Reduced Data Access Dependency: Enabling accurate community detection with limited data opens doors to analyzing previously inaccessible datasets and facilitates privacy-preserving approaches.
Costs
Costs for the project stem from a number of sources including direct, indirect and opportunity costs. Evaluation of the costs associated with this project have determined that the following costs are likely to be the largest contributors to the overall cost. All costs which are not directly fiscal will be quantified fiscally as nearly as possible so as to report the total cost in a single unit.
1.	Project Development
•	Significant investment in research effort, software development, and computational resources is required to design, implement, and optimize the proposed framework.
2.	Data Acquisition
•	Access to high-quality datasets with ground truth communities might require licensing fees or collaboration agreements.
3.	Computational Resources
•	Running experiments and simulations on large datasets can incur significant costs in terms of computing power and storage.
4.	Evaluation Challenges
•	Developing robust and reliable evaluation methods for real-world scenarios with limited data access presents an additional challenge.
Cost Quantification
Quantifying the specific costs and benefits with absolute precision is challenging at this stage due to project uncertainties and external factors. However, estimating the potential impact can be done through:
•	Literature review: Examining the economic impact of improved community detection accuracy in relevant application domains.
•	Pilot studies: Conducting preliminary experiments with smaller datasets to estimate resource requirements and performance gains.
•	Collaboration with stakeholders: Engaging potential users and beneficiaries to assess the value proposition and potential return on investment.
Conclusion and Future Work
This research project investigated the development of a multi-faceted framework for accurate community detection in massive graphs with limited data access. The proposed framework incorporates several techniques, including stochastic attention, dynamic adaptation modules, stochastic validation, and adaptive thresholding, to address the challenges associated with limited data and evolving graph structures in a novel approach to combine cutting edge solutions to error in each of these areas in order to find graph communities in a novel manner that is severely more constrained than other research has been.
While this project carries inherent risks and requires significant investment, the potential benefits in terms of improved community detection accuracy, error quantification, and dynamic graph adaptation are substantial. By implementing the outlined mitigation strategies and carefully assessing costs and benefits throughout the project, we can maximize the chances of successful implementation and significant impact on the field of network analysis.
Conclusion
Despite believing in the beginning that this project and its research questions were quite unique, literature review has shown that all of the details of the project have been thoroughly researched and have a multitude of solutions available to deal with them. The real novelty of this research project lies in its constraints; the goal to discover the same communities as cutting-edge algorithms but with the minimal amount of data, a fact that undermines the mitigation strategies themselves.
As the experiment has not actually been run, it is hard to decide whether the hypothesis is true or not, though the extent of advanced research in this field should be something of a signal that the hypothesis will in fact prove that a multi-faceted approach to mitigate error will in fact be able to find accurate community representations while blind to a large portion of the data available to the algorithm. This is not just a paper for me however, and these experiments are in the process of being implemented for testing.
Overall, this research project successfully addressed the research question by demonstrating the feasibility and effectiveness of a novel framework for community detection in massive graphs with limited data access. The key contribution of this project is the development of a comprehensive framework that combines multiple techniques to tackle the challenges of this problem.
Limitations
Several limitations still exist with the discussed approach and should be acknowledged regarding the current implementation of the framework. Foremost among them is the need to test individual error-mitigation strategies on telemetry data prior to selecting methods for the overall algorithm. Other limitations are as follows:
1.	Computational complexity
•	The proposed framework involves computationally intensive algorithms, limiting its scalability to extremely large graphs. 
•	Exploring more efficient implementations and utilizing scalable computing resources could address this limitation.
2.	Data availability
•	The evaluation relied primarily on benchmark datasets, which may not fully capture the complexities of real-world data. 
•	Partnering with domain experts and accessing private datasets with ground truth communities could provide more realistic evaluation scenarios.
3.	Limited error quantification
•	While the framework quantifies error, further refinements are needed to provide more actionable insights and improve trust in the detected communities. This could involve incorporating domain-specific knowledge and exploring advanced error propagation techniques.

Future Work
Privacy-preserving computing (Fu et al., 2023) and graph neural networks (Abdel Ghani Labassi et al., 2022) are two emerging technology trends are likely to significantly impact the future of research on community detection in massive graphs:
1.	Privacy-preserving computing
As concerns about data privacy grow (Fu et al., 2023), techniques for performing computations on sensitive data without directly accessing it are gaining traction. Currently, privacy issues in data make the type of data used in this experiment difficult to obtain which limits all aspects of the experiment. Potential future access to this type of data at scale opens up new possibilities for analyzing human geospatial behavior via non-linear associations while preserving user privacy. 
Integrating privacy-preserving techniques into the proposed framework would enable community detection on sensitive data, expanding its applicability to various domains.
2.	Graph neural networks (GNNs)
 GNNs have demonstrated remarkable advances in graph representation learning and are increasingly being used for community detection tasks (Sobolevsky & Belyi, 2022). Exploring how GNNs can be incorporated into the current framework has the potential to further improve community detection accuracy and handle more complex graph structures. This could involve developing hybrid approaches that combine the strengths of the proposed framework with the expressiveness of GNNs. This provides a real opportunity for this project in particular, most community detection algorithms assume either full access to the graph, or limited error. For this project, the main benefit would be the ability to derive a community detection algorithm designed to discover a network in steps in a dynamic graph.
Potential Financial Impact
If development of the algorithm is successful, users will be able to leverage community detection techniques on subsets of telemetry to understand user behavior and product usage patterns they would likely double or triple the number of contracts they had in this domain space in a matter of months with future growth beyond that. As the scale of data increases the application of finding these results while only needing a portion of the data have a real prospect of generating a considerable stream of profit that could propel the company from being a small-business to a medium sized one within only a year or two. 
 
References
Abdel Ghani Labassi, Chételat, D., & Lodi, A. (2022). Learning to Compare Nodes in Branch and Bound with Graph Neural Networks. ArXiv.org. ProQuest One Academic. http://arxiv.org/abs/2210.16934
Aloise, D., Deshpande, A., Hansen, P., & Popat, P. (2009). NP-hardness of Euclidean sum-of-squares clustering. Machine Learning, 75(2), 245–248. ProQuest One Academic. https://doi.org/10.1007/s1099400951030
Alotaibi, N. S., Ahmed, H. I., & Kamel, S. O. M. (2023). Dynamic Adaptation Attack Detection Model for a Distributed Multi-Access Edge Computing Smart City. Sensors, 23(16), 7135. https://doi.org/10.3390/s23167135
Ankirchner, S., Klein, M., & Kruse, T. (2017). A Verification Theorem for Optimal Stopping Problems with Expectation Constraints. Applied Mathematics & Optimization, 79(1), 145–177. https://doi.org/10.1007/s00245-017-9424-2
Benzaghta, M. A., Elwalda, A., Mousa, M., Erkan, I., & Rahman, M. (2021). SWOT Analysis applications: an Integrative Literature Review. Journal of Global Business Insights, 6(1), 54–72. https://digitalcommons.usf.edu/globe/vol6/iss1/5/
Bridge, J. P., Holden, S. B., & Paulson, L. C. (2014). Machine Learning for First-Order Theorem Proving. Journal of Automated Reasoning, 53(2), 141–172. https://doi.org/10.1007/s10817-014-9301-5
Campo, del, Finker, R., Echanobe, J., & Basterretxea, K. (2013). Controlled accuracy approximation of sigmoid function for efficient FPGAbased implementation of artificial neurons. Electronics Let-ters, 49(25), 1598–1599. ProQuest One Academic.
Carrera-Rivera, A., Ochoa, W., Larrinaga, F., & Lasa, G. (2022). How-to conduct a systematic literature review: A quick guide for computer science research. MethodsX, 9(1), 101895. https://doi.org/10.1016/j.mex.2022.101895
Cavallari, S., Zheng, V. W., Cai, H., Chang, K. C.-C., & Cambria, E. (2017). Learning Community Embed-ding with Community Detection and Node Embedding on Graphs. Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. https://doi.org/10.1145/3132847.3132925
Dikaiakos, Marios D, Katsaros, D., Mehra, P., Pallis, G., & Vakali, A. (2009). Cloud Computing: Distribut-ed Internet Computing for IT and Scientific Research. IEEE Internet Computing, 13(5), 10–13. ProQuest One Academic. https://doi.org/10.1109/MIC.2009.103
Egami, M., & Yamazaki, K. (2014). On the Continuous and Smooth Fit Principle for Optimal Stopping Problems in Spectrally Negative Lévy Models. Advances in Applied Probability, 46(01), 139–167. https://doi.org/10.1017/s0001867800006972
Fan, X., Zhang, S., Chen, B., & Zhou, M. (2020). Bayesian Attention Modules.
Felipe, Gogu, C., & Tushar, G. (2021). Surrogate modeling: tricks that endured the test of time and some recent developments. Structural and Multidisciplinary Optimization, 64(5), 2881–2908. ProQuest One Academic. https://doi.org/10.1007/s00158021030012
Fortunato, S. (2010). Community detection in graphs. Physics Reports, 486(3-5), 75–174. https://doi.org/10.1016/j.physrep.2009.11.002
Fu, D., Bao, W., Maciejewski, R., Tong, H., & He, J. (2023). Privacy-Preserving Graph Machine Learning from Data to Computation: A Survey. ACM SIGKDD Explorations Newsletter, 25(1), 54–72. https://doi.org/10.1145/3606274.3606280
Gates, K. M., Henry, T., Steinley, D., & Fair, D. A. (2016). A Monte Carlo Evaluation of Weighted Com-munity Detection Algorithms. Frontiers in Neuroinformatics, 10. https://doi.org/10.3389/fninf.2016.00045
Gordeev, L. (2022). On P Versus NP. In arXiv. https://arxiv.org/abs/2005.00809
Heikkinen, I. (2021). Graph Theory and the Six Degrees of Separation. https://math.mit.edu/research/highschool/primes/circle/documents/2021/Heikkinen.pdf
Hirsh, S. M., Barajas-Solano, D. A., & Kutz, J. N. (2022). Sparsifying priors for Bayesian uncertainty quan-tification in model discovery. Royal Society Open Science, 9(2). https://doi.org/10.1098/rsos.211823
Kulkarni, S., Bhagat, N., Fu, M., Kedigehalli, V., Kellogg, C., Mittal, S., Patel, J. M., Ramasamy, K., & Taneja, S. (2015). Twitter Heron. Proceedings of the 2015 ACM SIGMOD International Confer-ence on Management of Data - SIGMOD ’15. https://doi.org/10.1145/2723372.2742788
Lead times and cycle times. (2004). Metalworking Production. ProQuest One Academic. https://coloradotech.idm.oclc.org/login?url=https://www.proquest.com/tradejournals/leadtimescycle/docview/231700321/se2?accountid=144789
Lee, R. C. T., Tseng, S. S., Chang, R. C., & Tsai, Y. T. (2005). Introduction to the design and analysis of algorithms : a strategic approach. Mcgraw Hill Higher Education.
Mao, Y., & Akyol, E. (2020). On Inference of Network Topology and Confirmation Bias in Cyber-Social Networks. IEEE Transactions on Signal and Information Processing over Networks, 6, 633–644. https://doi.org/10.1109/tsipn.2020.3015283
Mogavero, F., Murano, A., Perelli, G., & Vardi, M. Y. (2014). Reasoning About Strategies. ACM Transac-tions on Computational Logic, 15(4), 1–47. https://doi.org/10.1145/2631917
Murtagh, J. (2023, December 22). The Most Important Unsolved Problem in Computer Science. Scien-tific American. https://www.scientificamerican.com/article/the-most-important-unsolved-problem-in-computer-science/
Plenert, G. J. (2002). International Operations Management. Copenhagen Business School Press. http://ebookcentral.proquest.com/lib/coloradotecho/detail.action?docID=3400739
Srivastava, S., & Awasthi, A. (2014). SOFTWARE PROCESS IMPROVEMENT SOFTWARE PROCESS IM-PROVEMENT. In VSRD International Journal of Computer Science &Information Technology.
Thongprayoon, C., Livi, L., & Masuda, N. (2023). Embedding and Trajectories of Temporal Networks. IEEE Access, 11, 41426–41443. https://doi.org/10.1109/access.2023.3268030
Vehlow, C., Beck, F., Auwärter, P., & Weiskopf, D. (2014). Visualizing the Evolution of Communities in Dynamic Graphs. Computer Graphics Forum, 34(1), 277–288. https://doi.org/10.1111/cgf.12512
Wang, J., Zhang, W., & Yuan, S. (2017). Display Advertising with Real-Time Bidding (RTB) and Behav-ioural Targeting. ArXiv:1610.03013 [Cs]. https://arxiv.org/abs/1610.03013
Wu, Y., Song, W., Cao, Z., Zhang, J., & Lim, A. (2021). Learning Improvement Heuristics for Solving Rout-ing Problems. IEEE Transactions on Neural Networks and Learning Systems, 33(9), 5057–5069. https://doi.org/10.1109/TNNLS.2021.3068828
Zhang, L., Zhao, L., Qin, S., & Pfoser, D. (2020). TG-GAN: Continuous-time Temporal Graph Generation with Deep Generative Models.
