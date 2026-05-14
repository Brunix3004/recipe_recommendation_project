# Semester Group Assignment Brief

## Title

Semester Project: Domain Discovery, Recommendation, and Graph Intelligence

## Purpose

This assignment is designed to run across the full 14-week cycle of the course.
It is not a single-model project.
It is a cumulative technical project that forces students to practice:

- data acquisition
- schema design
- feature engineering
- dimensionality reduction
- clustering
- recommendation or ranking
- graph analytics
- reproducible pipelines
- technical reporting and final defense

The project must be technical first.
The written report matters, but the strongest submissions will be judged by the quality of the data pipeline, the rigor of the experiments, and the reproducibility of the artifacts.

## Core Project Shape

Every team must build a project with the following layers:

1. `catalog layer`
- the main entity table
- examples: posts, profiles, recipes, products, songs, businesses, papers, events, courses

2. `feature layer`
- numeric, categorical, text, temporal, or graph-derived features

3. `interaction or co-occurrence layer`
- examples: clicks, ratings, saves, baskets, reviews, shared tags, co-mentions, follows, lists, sessions, playlists

4. `graph layer`
- examples: item-item, hashtag-hashtag, skill-skill, business-business, paper-paper, user-item projection, career transition graph

5. `pipeline layer`
- a reproducible build path from raw data to processed data to analysis artifacts

Projects that stop at basic supervised learning will be considered incomplete for this course.

## Allowed Project Tracks

Teams may choose one of the following:

### Track A: Public Dataset Project

Use one or more public datasets from Kaggle, government/open data portals, official research releases, or platform-sanctioned exports.

Examples:

- news recommendation
- local business analytics
- recipe recommendation
- niche e-commerce recommendation
- academic-paper intelligence
- event discovery
- book or game recommendation

### Track B: Build-Your-Own Dataset

Create a dataset from public websites or APIs, subject to legality and platform rules.
This track is strongly encouraged when the team wants a more original domain.

Examples:

- startup tools
- local cultural events
- podcast catalogs
- indie games
- open course catalogs
- scientific software libraries
- board game communities

### Track C: Instagram Content Intelligence

Use Instagram data only through approved means:

- data exports from consenting participants
- business or creator account analytics
- platform-approved API access

Suggested problem framing:

- cluster content styles
- analyze engagement patterns
- recommend hashtags, content themes, or posting strategies
- build hashtag or mention graphs

### Track D: LinkedIn Career and Skill Intelligence

Use LinkedIn data only through approved means:

- data exports from consenting participants
- page-admin analytics if the team has valid access
- approved API access where available

Suggested problem framing:

- normalize and cluster career paths
- build skill graphs
- rank skills, roles, or career transitions
- recommend skill-development directions or role families

### Track E: Cross-Platform Project

Combine multiple sources when the team can justify the match.

Examples:

- Instagram plus LinkedIn personal-brand analytics
- local-business reviews plus social media content
- publication metadata plus citation graph plus author skills

This track is harder and should only be approved if the team can explain the data alignment clearly.

## Strongly Recommended Project Types

The best fit for this course is a `discovery and recommendation system`.

That means the final output should answer questions such as:

- Which items are similar?
- Which items belong to the same latent segment?
- Which items should be recommended next?
- Which entities are central in the graph?
- Which changes in content or behavior are associated with better outcomes?

## Forbidden or Weak Project Choices

The following are not acceptable:

- Titanic
- Iris
- Penguins
- MNIST used as a standard teaching exercise
- any tiny classroom dataset with no meaningful engineering challenge
- any project with only one flat CSV and no pipeline
- any project with only screenshots and no reproducible code

The following are technically possible but weak for this course unless extended substantially:

- plain salary prediction with one table only
- plain house-price regression with one table only
- any project without either an interaction layer or a graph layer

## Team Size

Recommended team size:

- `3 to 5` students per team

Each team must assign at least these roles, even if one student covers more than one:

- data engineering lead
- modeling and evaluation lead
- reporting and presentation lead

## Dataset Rules

Every team must satisfy all of the following:

1. Use a dataset that is non-trivial in size, structure, or preprocessing difficulty.
2. Provide a data dictionary.
3. Document all source URLs and provenance.
4. Explain why the chosen dataset supports the second half of the course, not just the first half.
5. Avoid trivial benchmark datasets.
6. If the dataset is too large, define a justified working subset.
7. If multiple sources are combined, document the matching logic.

## Ethics and Platform Rules

These rules are mandatory.

### General

- No unauthorized scraping of private or access-controlled data.
- No use of personal data without clear participant consent.
- No redistribution of raw personal data collected from participants.
- Sensitive identifiers must be anonymized or hashed in processed artifacts unless explicit permission is documented.

### Instagram

Instagram-based projects are allowed only when the data comes from:

- consenting participant exports
- consenting business or creator account analytics
- platform-approved API access

Teams must not scrape private profiles or bypass platform restrictions.

### LinkedIn

LinkedIn-based projects are allowed only when the data comes from:

- consenting participant exports
- page-admin data with valid access
- platform-approved API access

Teams must not scrape personal profiles without permission.

### Required Ethics Note

Each milestone must include a short section named `Ethics and Access Note` that states:

- where the data came from
- why the team is allowed to use it
- what personal-data risks exist
- how those risks were reduced

## Required Repository Structure

Each team repository must include a structure equivalent to this:

```text
project/
  data/
    raw/
    interim/
    processed/
  notebooks/
  src/
  reports/
  artifacts/
  README.md
  requirements.txt or pyproject.toml
```

The exact names may vary, but the separation between raw, processed, code, and reports must be explicit.

## Required Technical Artifacts

Each team must produce:

1. one ingestion script or pipeline
2. one processed dataset directory
3. one data dictionary
4. one feature-building script or notebook
5. one evaluation script
6. one graph-construction script
7. one runbook explaining how to reproduce the outputs
8. one final demo artifact
- this may be a notebook, CLI, small dashboard, or small API

Notebook-only projects are not enough.
At least part of the workflow must be executable from scripts or commands.

## Mandatory Milestones

The project has partial deliverables in Weeks 3, 5, 7, 10, 12, and 14.

### Week 3: Dataset Charter and Processed Dataset V1

Objective:

- prove that the team has a valid domain, a valid data source, and a reproducible first dataset build

Required deliverables:

1. one project proposal
- domain
- problem statement
- expected product question
- why the dataset is suitable for the course

2. one source inventory
- source URLs
- licenses or access conditions
- raw file formats
- estimated size

3. one schema draft
- entity tables
- keys
- expected joins

4. one processed dataset V1
- at least one cleaned table saved to disk

5. one data dictionary draft

6. one scale analysis
- rows
- columns
- missingness
- sparsity or memory estimate

7. one ethics and access note

Technical expectation:

- the ingestion must run from a documented command

### Week 5: Representation and Dimensionality Report

Objective:

- move from raw tables to meaningful representations

Required deliverables:

1. one feature matrix or matrices
- numeric
- text
- categorical encoding
- temporal features if relevant

2. one dimensionality-reduction report
- PCA and/or SVD
- optional t-SNE for visualization

3. one comparison table
- explained variance or retained energy
- reconstruction error where appropriate

4. one visualization set
- at least two meaningful plots

5. one short technical interpretation
- what was learned about dimensionality, redundancy, or feature quality

Technical expectation:

- the feature-building pipeline must be reproducible

### Week 7: Clustering and Validation Report

Objective:

- segment the domain and validate whether the segmentation is meaningful

Required deliverables:

1. one clustering experiment with K-means
2. optional: one clustering experiment with DBSCAN or another justified density method
3. one validation table
- silhouette
- inertia or density-related metrics
- interpretation limits

4. one cluster-profile analysis
- what characterizes the clusters

5. one failure analysis
- what did not cluster well and why

Technical expectation:

- the team must show parameter sweeps, not only one hand-picked result

### Week 10: Recommendation, Ranking, or Predictive Decision Engine

Objective:

- connect the technical work to a decision or ranking task

Required deliverables:

1. one baseline system
- content-based, popularity-based, or another simple baseline

2. one stronger system
- collaborative filtering
- matrix factorization
- hybrid ranking
- predictive ranking
- or another justified advanced model

3. one offline evaluation report
- metrics
- candidate-pool definition
- comparison against baselines

4. one error analysis
- strong cases
- failure cases

5. one explanation of whether the project is:
- recommendation
- ranking
- prediction
- segmentation feeding ranking

Technical expectation:

- teams must explain their evaluation protocol clearly
- if they claim a hybrid system, they must document the data alignment

### Week 12: Graph Analytics and Centrality Report

Objective:

- formalize the graph induced by the domain and use it for structural analysis

Required deliverables:

1. one graph definition
- nodes
- edges
- weights
- directionality

2. one graph-construction script

3. one graph report
- connected components
- degree or weighted degree
- centrality or PageRank

4. one comparison section
- graph ranking versus popularity or model-based ranking

5. one interpretation note
- what graph structure means in the domain

Technical expectation:

- graph definition choices must be justified

### Week 14: Final Integrated Delivery and Defense

Objective:

- deliver the complete system and defend its coherence

Required deliverables:

1. one final technical report
2. one reproducible repository
3. one runbook
4. one final presentation
5. one final demo artifact
6. one monitoring or operationalization plan
7. one limitations and future-work section

Technical expectation:

- the project must run from documented steps
- the team must show final processed artifacts and outputs, not only slides

## Final Report Structure

The final report should contain:

1. problem statement
2. domain context
3. dataset sources and access conditions
4. schema and data dictionary
5. preprocessing and feature engineering
6. dimensionality and representation analysis
7. clustering analysis
8. recommendation or ranking system
9. graph analytics
10. evaluation protocol
11. pipeline and reproducibility
12. ethics and limitations
13. final conclusions


## Detailed Evaluation Criteria

### Data Engineering

The team will be evaluated on:

- correctness of ingestion
- schema clarity
- quality of processed data
- quality of data dictionary
- handling of missing values, duplicates, and joins

### Modeling and Analysis

The team will be evaluated on:

- representation quality
- parameter justification
- baseline comparisons
- metric rigor
- interpretation quality

### Graph and Advanced Layer

The team will be evaluated on:

- validity of graph construction
- correctness of centrality or PageRank analysis
- connection between graph results and product question

### Engineering and Reproducibility

The team will be evaluated on:

- repo structure
- clear commands
- saved artifacts
- rerunnable workflow
- evidence that the project does not rely on hidden notebook state

### Final Defense

The team will be evaluated on:

- coherence between objective, data, model, and evaluation
- clarity of technical decisions
- honesty about limitations
- quality of oral defense and technical answers

## Suggested Topic Proposals

These are strong examples, not an exhaustive list.

### Proposal 1: News Recommendation and Topic Graph

Build a system that:

- represents articles using text features
- models click behavior or reading behavior
- recommends articles
- builds an entity or topic graph

### Proposal 2: Local Business Discovery Engine

Build a system that:

- models businesses, reviews, categories, and neighborhoods
- clusters businesses
- recommends businesses
- builds business-business or category graphs

### Proposal 3: Recipe Recommendation and Ingredient Graph

Build a system that:

- represents recipes from ingredients and instructions
- clusters recipe styles
- recommends recipes
- builds an ingredient graph

### Proposal 4: Instagram Content Intelligence

Build a system that:

- represents posts and captions
- predicts or ranks engagement potential
- clusters content strategies
- builds hashtag or mention graphs

### Proposal 5: LinkedIn Career and Skill Intelligence

Build a system that:

- normalizes roles and skills
- clusters professional profiles or role families
- ranks important skills
- builds skill or career-transition graphs

### Proposal 6: Niche E-commerce Recommender

Build a system that:

- models products, metadata, reviews, and interactions
- clusters products
- recommends products
- builds product co-occurrence or co-purchase graphs

### Proposal 7: Academic Paper Discovery System

Build a system that:

- represents papers using titles, abstracts, topics, and citations
- clusters research themes
- recommends papers
- ranks papers, authors, or venues through graph methods

## Project Approval Rule

No project is approved until the instructor confirms that:

1. the dataset is non-trivial
2. the data source is allowed
3. the second-half course topics are feasible on the chosen data
4. the team has a credible build plan

Teams whose topic is too weak at Week 3 may be required to pivot.

## Minimum Technical Standard for Passing

To pass the project, a team must show all of the following:

- a real processed dataset
- at least one rigorous feature representation
- at least one clustering experiment
- at least one ranking or recommendation experiment
- at least one graph analysis
- a reproducible build path

If any of these layers is missing, the project may be graded as incomplete.

## Final Note to Students

Do not choose a project because it looks easy.
Choose a project whose data structure can survive the entire semester.

The right question is not:

> Can we train one model on this data?

The right question is:

> Can we build a coherent data product from ingestion to ranking to graph analysis to final defense?


# Semester Group Assignment Rubric

## Purpose

This rubric evaluates the semester group assignment described in:

- `big_data_course_content/semester_group_assignment_brief.md`

The assignment is a cumulative technical project.
Teams are expected to build a coherent data product from ingestion to representation, clustering, recommendation or ranking, graph analysis, reproducible delivery, and final defense.

The grading priority is:

1. ability to defend decisions under questioning
2. technical correctness
3. reproducibility
4. interpretation and discussion
5. depth of analysis
6. coherence across the full project

Each assessed deliverable is graded on a `20-point` scale.
If the instructor combines these deliverables into one final project grade, the final defense and question-response component should receive the heaviest external weight because the project must be defended as an integrated technical system, not only submitted as files.

Although grades may be assigned individually, the project must be defended by the group during the scheduled presentation.
The group is responsible for defending the submitted work, the presentation, and the technical decisions behind the project.
If the group does not defend the work in the presentation, the deliverable receives no grade.

---

## Scoring Model

Each deliverable below has a maximum score of `20 points`.
The point values inside each section sum to `20`.

| Assessed deliverable | Maximum score |
| --- | ---: |
| Week 3: Dataset Charter and Processed Dataset V1 | 20 |
| Week 5: Representation and Dimensionality Report | 20 |
| Week 7: Clustering and Validation Report | 20 |
| Week 10: Recommendation, Ranking, or Predictive Decision Engine | 20 |
| Week 12: Graph Analytics and Centrality Report | 20 |
| Week 14: Final Integrated Delivery | 20 |
| Final Technical Defense and Question Response | 20 |

Minimum completeness rule:

- A project missing a real processed dataset, a rigorous feature representation, a clustering experiment, a recommendation/ranking experiment, a graph analysis, or a reproducible build path may be capped at a passing-minimum or marked incomplete, even if other sections are polished.

---

## Performance Levels

Use these levels consistently across all components.

| Level | Meaning |
| --- | --- |
| Excellent | Technically correct, reproducible, well justified, deeply interpreted, and clearly connected to the product question. |
| Good | Mostly correct and reproducible, with reasonable interpretation and only minor gaps. |
| Satisfactory | Basic deliverable is present, but justification, rigor, reproducibility, or interpretation is shallow. |
| Weak | Deliverable is incomplete, hard to reproduce, weakly connected to the project, or mostly descriptive. |
| Missing | Required artifact or explanation is absent. |

---

## Defense Question Emphasis

Across all deliverables, defense questions are graded primarily on the quality of the team's reasoning.
The instructor evaluates why decisions were made more heavily than simply what was done or how it was implemented.
Strong answers connect evidence to tradeoffs, alternatives considered, assumptions, limitations, and the product question.

Teams are expected to take a critical view of their own work.
They should identify what worked, what did not work, what could have been better, what should not be overclaimed, and what they would change with more data, more time, or a different technical constraint.
This critical self-assessment is part of the defense-question score for every deliverable.

---

# Week 3 Rubric: Dataset Charter and Processed Dataset V1

Total: `20 points`

Objective:

- prove that the team has a valid domain, a valid data source, and a reproducible first dataset build

## Required Deliverables and Rubric

| Deliverable | Points | Excellent evidence |
| --- | ---: | --- |
| Project proposal | 1.25 | Domain, problem statement, expected product question, and course fit are specific and technically feasible. |
| Source inventory | 1.25 | Source URLs, licenses/access conditions, raw formats, estimated size, and provenance are complete and auditable. |
| Schema draft | 1.25 | Entity tables, keys, joins, grain, and relationship assumptions are clear. |
| Processed dataset V1 | 1.5 | At least one cleaned table is saved to disk and can be regenerated. |
| Data dictionary draft | 1.0 | Columns, types, meanings, units, and known quality issues are documented. |
| Scale analysis | 1.0 | Rows, columns, missingness, sparsity, memory estimate, and subset justification are reported. |
| Ethics and Access Note | 1.25 | Data permission, personal-data risks, and mitigation steps are explicit. |
| Reproducible ingestion command | 1.5 | The ingestion path runs from a documented command without hidden notebook state. |
| Defense questions | 10.0 | The team can orally defend source legitimacy, schema design, data quality, scale, ethics, and how the dataset supports later course layers. |

## Defense Questions

Defense answers are evaluated more on why the team made each decision than on merely describing what was done or how it was implemented.
Strong answers include critical self-assessment: limitations, alternatives, weak assumptions, what could have been better, and what should not be overclaimed.

- What is the exact product question, and why does this dataset support it?
- What is the grain of each table, and what are the key joins?
- Which source is highest risk, legally or technically, and how did you reduce that risk?
- What evidence shows that the processed dataset was generated reproducibly?
- What later project layers are feasible from this dataset: representation, clustering, recommendation/ranking, and graph analysis?

## Interpretation and Discussion Expectations

Strong Week 3 submissions explain why the dataset can support the second half of the course.
They do not only say "we have data."
They explain what can become an interaction layer, a graph layer, and a recommendation or ranking task.

## Common Deductions

- vague product question
- missing source permissions
- no clear entity grain
- processed data created manually without a reproducible command
- data dictionary that only repeats column names
- ethics note that says "no issue" without analysis

---

# Week 5 Rubric: Representation and Dimensionality Report

Total: `20 points`

Objective:

- move from raw tables to meaningful representations

## Required Deliverables and Rubric

| Subject | Excellent evidence | Max points |
| --- | --- | ---: |
| Feature matrix or matrices | Numeric, text, categorical, temporal, or graph-derived features are built with clear preprocessing choices. | 2 |
| Dimensionality-reduction report | PCA and/or SVD are correctly applied; optional t-SNE is clearly labeled as visualization, not compression. | 1.5 |
| Comparison table | Explained variance, retained energy, reconstruction error, or equivalent metrics are reported and interpreted correctly. | 1.5 |
| Visualization set | At least two meaningful plots support the analysis and are not merely decorative. | 1 |
| Technical interpretation | The team explains dimensionality, redundancy, feature quality, representation limits, and what each method preserves. | 3 |
| Reproducible feature-building pipeline | Feature generation can be rerun from scripts or documented commands. | 1 |
| Defense questions | The team can defend preprocessing choices, representation quality, dimensionality metrics, visualization limits, and why the representation supports the product question. Focus is more on the why rather than the what. | 10 |

## Defense Questions

Defense answers are evaluated more on why the team made each decision than on merely describing what was done or how it was implemented.
Strong answers include critical self-assessment: limitations, alternatives, weak assumptions, what could have been better, and what should not be overclaimed.

- Why did you choose this representation instead of a simpler one?
- What preprocessing choice most changed the representation?
- What does PCA, SVD, or the chosen dimensionality method optimize?
- What should not be concluded from your visualizations?
- What evidence supports your interpretation beyond the plot?

## Interpretation and Discussion Expectations

This milestone must answer:

- Which representation is most appropriate for the product question?
- What preprocessing choices most affected the representation?
- What dimensionality-reduction method was used, and what objective does it optimize?
- What evidence supports the interpretation beyond a visual plot?
- What are the limitations of the chosen representation?

Deep analysis should distinguish:

- PCA/SVD as compression or reconstruction-oriented methods
- text vectors as sparse lexical representations
- learned embeddings as objective-dependent representations
- t-SNE as exploratory visualization, not a production feature space

## Common Deductions

- using t-SNE as proof of clusters
- reporting plots without explaining what structure is preserved
- no reconstruction or retained-energy discussion
- feature leakage or unexplained preprocessing
- no comparison against a simple baseline representation

---

# Week 7 Rubric: Clustering and Validation Report

Total: `20 points`

Objective:

- segment the domain and validate whether the segmentation is meaningful
- DBSCAN and hierarchical clustering parts are optional

## Required Deliverables and Rubric

| Deliverable | Points | Excellent evidence |
| --- | ---: | --- |
| K-means experiment | 3 | K values are explored, initialization is controlled, and assumptions are discussed. |
| DBSCAN or justified density method | 0 | Parameter choices are explored and linked to density assumptions. |
| Validation table | 2.0 | Silhouette, inertia or density-related metrics, and interpretation limits are reported clearly. |
| Cluster-profile analysis | 2.0 | Clusters are characterized with domain features, examples, and meaningful differences. |
| Failure analysis | 1.5 | The team explains what did not cluster well and why. |
| Parameter sweeps | 1.5 | Results are not based on one hand-picked run; sensitivity is shown. |
| Defense questions | 10.0 | The team can defend representation choice, clustering assumptions, parameter sensitivity, validation limits, cluster meaning, and failure cases. |

## Defense Questions

Defense answers are evaluated more on why the team made each decision than on merely describing what was done or how it was implemented.
Strong answers include critical self-assessment: limitations, alternatives, weak assumptions, what could have been better, and what should not be overclaimed.

- Why did you cluster this representation?
- What assumptions does K-means make, and where do they fail here?
- How did DBSCAN or the density method change the interpretation?
- Which cluster is most defensible, and which is weakest?
- What validation evidence supports the cluster profiles?

## Interpretation and Discussion Expectations

Strong submissions do not ask only "which metric is highest?"
They discuss whether the clusters make sense for the domain and product question.

They should explain:

- which representation was clustered
- why that representation was chosen
- whether clusters are stable under parameter changes
- what each cluster means operationally
- what claims the validation metrics do and do not support

## Common Deductions

- one clustering run with no sweep
- cluster labels reported with no profile
- using t-SNE islands as cluster validation
- no discussion of noise points or outliers
- no connection between clusters and the final product goal

---

# Week 10 Rubric: Recommendation, Ranking, or Predictive Decision Engine

Total: `20 points`

Objective:

- connect the technical work to a decision or ranking task

## Required Deliverables and Rubric

| Deliverable | Points | Excellent evidence |
| --- | ---: | --- |
| Baseline system | 1.5 | A simple baseline is implemented and justified, such as popularity-based or content-based ranking. |
| Stronger system | 2.0 | Collaborative filtering, matrix factorization, hybrid ranking, predictive ranking, or another advanced model is correctly implemented. |
| Offline evaluation report | 2.5 | Metrics, candidate-pool definition, train/test logic, and baseline comparison are explicit. |
| Error analysis | 1.5 | Strong cases and failure cases are examined with examples and domain interpretation. |
| Task framing explanation | 1.0 | The team clearly states whether the system is recommendation, ranking, prediction, or segmentation feeding ranking. |
| Data alignment documentation | 1.5 | Interaction, catalog, feature, and model inputs are aligned without hidden assumptions. |
| Defense questions | 10.0 | The team can defend the task framing, baseline, stronger model, candidate pool, metrics, error cases, and data alignment. |

## Defense Questions

Defense answers are evaluated more on why the team made each decision than on merely describing what was done or how it was implemented.
Strong answers include critical self-assessment: limitations, alternatives, weak assumptions, what could have been better, and what should not be overclaimed.

- What is the exact candidate pool?
- Why is your baseline fair?
- What does your stronger system improve, and where does it fail?
- What metric did you optimize, and why is it appropriate for the product question?
- How did you prevent leakage across users, items, time, or interactions?

## Interpretation and Discussion Expectations

The report must answer:

- What is the decision being supported?
- What is the unit being ranked or recommended?
- What is the candidate pool?
- What does a correct recommendation mean in this domain?
- Which baseline must the advanced method beat?
- Where does the model fail and why?

Depth is shown through comparison, not model complexity alone.
A complex model that is not better than a simple baseline must be discussed honestly.

## Common Deductions

- no baseline
- metrics reported without candidate-pool definition
- train/test leakage
- no examples of good and bad recommendations
- claiming a hybrid model without explaining data alignment
- optimizing a metric that does not match the product question

---

# Week 12 Rubric: Graph Analytics and Centrality Report

Total: `20 points`

Objective:

- formalize the graph induced by the domain and use it for structural analysis

## Required Deliverables and Rubric

| Deliverable | Points | Excellent evidence |
| --- | ---: | --- |
| Graph definition | 2.0 | Nodes, edges, weights, directionality, and graph grain are precise and justified. |
| Graph-construction script | 1.5 | The graph can be regenerated from processed data. |
| Graph report | 2.0 | Connected components, degree or weighted degree, centrality, PageRank, or related measures are computed correctly. |
| Comparison section | 1.5 | Graph ranking is compared against popularity, model-based ranking, or another baseline. |
| Interpretation note | 2.0 | The team explains what graph structure means in the domain and what it does not mean. |
| Validity checks | 1.0 | The team checks edge sparsity, isolated nodes, component structure, and sensitivity to graph-definition choices. |
| Defense questions | 10.0 | The team can defend node and edge definitions, weights, directionality, centrality meaning, comparison baselines, and graph limitations. |

## Defense Questions

Defense answers are evaluated more on why the team made each decision than on merely describing what was done or how it was implemented.
Strong answers include critical self-assessment: limitations, alternatives, weak assumptions, what could have been better, and what should not be overclaimed.

- What does a node represent, and what does an edge represent?
- Why is the graph weighted or unweighted?
- Why is the graph directed or undirected?
- What does centrality or PageRank mean in this domain?
- How does graph ranking differ from popularity or model-based ranking?

## Interpretation and Discussion Expectations

Strong graph analysis explains why this graph is the right representation.
It should answer:

- What does an edge mean?
- Why is the edge weighted or unweighted?
- Why is the graph directed or undirected?
- What does centrality mean in this domain?
- How does graph ranking differ from popularity or model ranking?

## Common Deductions

- graph built only for decoration
- unclear node or edge meaning
- no justification of edge threshold
- centrality reported without interpretation
- no comparison to a non-graph ranking
- no discussion of disconnected components

---

# Week 14 Rubric: Final Integrated Delivery

Total: `20 points`

Objective:

- deliver the complete system and prepare it for defense

## Required Deliverables and Rubric

| Deliverable | Points | Excellent evidence |
| --- | ---: | --- |
| Final technical report | 1.5 | Report follows the required structure and integrates all project layers coherently. |
| Reproducible repository | 1.5 | Repo structure is clean; raw, processed, code, reports, and artifacts are separated. |
| Runbook | 1.5 | A reader can rebuild outputs from documented steps and commands. |
| Final presentation | 1.0 | Slides communicate the system, evidence, and decisions clearly without hiding technical details. |
| Final demo artifact | 1.5 | Notebook, CLI, dashboard, or API works and shows a meaningful end-to-end use case. |
| Monitoring or operationalization plan | 1.0 | Plan discusses serving assumptions, model/data drift, retraining, logging, and failure modes. |
| Limitations and future work | 1.0 | Limitations are honest, specific, and tied to evidence from the project. |
| Final processed artifacts and outputs | 1.0 | Final data, models, reports, figures, and outputs are saved and easy to inspect. |
| Defense questions | 10.0 | The team can defend how the final report, repo, runbook, demo, limitations, and monitoring plan form one coherent system. |

## Defense Questions

Defense answers are evaluated more on why the team made each decision than on merely describing what was done or how it was implemented.
Strong answers include critical self-assessment: limitations, alternatives, weak assumptions, what could have been better, and what should not be overclaimed.

- How do we rerun the full project from raw or staged data to final outputs?
- What is the strongest result, and what evidence supports it?
- What is the weakest part of the system, and what would you improve first?
- What would you monitor if the system were run repeatedly?
- How do the data, representation, model, graph, and demo fit together?

## Final Report Alignment

The final report should include:

1. problem statement
2. domain context
3. dataset sources and access conditions
4. schema and data dictionary
5. preprocessing and feature engineering
6. dimensionality and representation analysis
7. clustering analysis
8. recommendation or ranking system
9. graph analytics
10. evaluation protocol
11. pipeline and reproducibility
12. ethics and limitations
13. final conclusions

## Interpretation and Discussion Expectations

The final delivery must synthesize the semester.
It should not read like separate weekly homework stapled together.

Strong submissions explain:

- why the data supports the product question
- how the representation choices affect downstream results
- what clustering revealed and what it failed to reveal
- why the recommendation or ranking system is credible
- what the graph analysis adds beyond the model
- how reproducibility was verified
- what limitations remain

## Common Deductions

- final report omits one major project layer
- demo does not run
- runbook depends on undocumented local paths
- final presentation overclaims results
- monitoring plan is generic and not tied to the system
- future work is vague

---

# Final Technical Defense and Question Response Rubric

Total: `20 points`

This is the heaviest component when the instructor combines deliverables into the final project grade.
Within this 20-point section, live question response is the most important evidence that the team actually understands and can defend the technical system they built.

The team must be ready for live questions about data, modeling, evaluation, graph construction, reproducibility, ethics, limitations, and product coherence.

## Defense Components

| Component | Points | Excellent evidence |
| --- | ---: | --- |
| System coherence defense | 2.0 | The team clearly explains how objective, data, schema, features, models, graph, evaluation, and demo form one coherent system. |
| Data and pipeline questions | 1.25 | The team can explain ingestion, cleaning, joins, data dictionary, missingness, scale, ethics, and reproducibility without handwaving. |
| Representation and interpretation questions | 1.25 | The team can defend feature choices, dimensionality methods, embeddings, visualizations, interpretation limits, and representation tradeoffs. |
| Modeling and evaluation questions | 1.25 | The team can justify baselines, advanced models, metrics, candidate pools, train/test design, and failure cases. |
| Graph analytics questions | 1.0 | The team can defend node/edge definitions, weights, directionality, centrality/PageRank interpretation, and graph ranking comparisons. |
| Deep analysis and discussion under follow-up questions | 2.25 | The team gives nuanced answers, recognizes tradeoffs, connects evidence across milestones, and avoids superficial claims. |
| Limitations, ethics, and operational-risk questions | 1.0 | The team gives honest, specific limitations; explains ethics/access constraints; and discusses monitoring or operational risks. |
| Individual question-response quality | 10.0 | Multiple team members answer clearly, technically, and directly; answers are grounded in artifacts and evidence. This is the highest-weighted line item in the defense. |

## Defense Questions

Defense answers are evaluated more on why the team made each decision than on merely describing what was done or how it was implemented.
Strong answers include critical self-assessment: limitations, alternatives, weak assumptions, what could have been better, and what should not be overclaimed.

The defense may include questions such as:

### Project Coherence

- What is the exact product question?
- What decision does your system support?
- Why does your dataset support that question?
- Which result is most important, and why?
- Which result should not be overinterpreted?

### Data and Reproducibility

- What is the grain of your main entity table?
- What are the primary keys and join keys?
- What were the highest-risk data quality issues?
- How do we rerun your pipeline from raw data to final artifacts?
- Which output would change if the raw data were refreshed?

### Representation and Interpretation

- Why did you choose this feature representation?
- What preprocessing choices most affected your results?
- What did PCA, SVD, embeddings, or visualizations reveal?
- What did they not reveal?
- What would be invalid to conclude from your plots?

### Clustering

- Why did you cluster this representation?
- How did K-means and DBSCAN differ in your project?
- Which parameters were most sensitive?
- What evidence supports your cluster interpretation?
- Which clusters are weak or unstable?

### Recommendation, Ranking, or Prediction

- What is your baseline?
- What is your stronger model?
- What is the candidate pool?
- What metric did you optimize, and why?
- Where does the model fail?
- What would happen if the data distribution changed?

### Graph Analytics

- What are nodes and edges?
- What does edge weight mean?
- Why is the graph directed or undirected?
- What does centrality mean in your domain?
- How does graph ranking differ from popularity or model ranking?

### Ethics and Operationalization

- Why are you allowed to use this data?
- What personal-data risks exist?
- How did you reduce those risks?
- What should be monitored if this system runs repeatedly?
- What is the biggest operational failure mode?

## Defense Performance Levels

| Level | Description |
| --- | --- |
| Excellent | Answers are precise, evidence-based, technically grounded, and integrated across the project. The team acknowledges uncertainty without collapsing the argument. |
| Good | Answers are mostly correct and connected to artifacts, with some gaps in nuance or cross-layer integration. |
| Satisfactory | Answers show basic familiarity but rely on generic explanations or shallow interpretation. |
| Weak | Answers are vague, inconsistent, or disconnected from the submitted artifacts. |
| Missing | The team cannot answer the question or no relevant artifact exists. |

## Defense Standards

Strong defense answers should:

- refer to specific files, tables, figures, scripts, metrics, or artifacts
- explain why a decision was made, not only what was done
- compare against at least one alternative
- identify limitations honestly
- separate evidence from speculation
- avoid claiming that a visualization alone proves a conclusion
- show that each team member understands more than their narrow task

Weak defense answers usually:

- describe tools without explaining objectives
- say "the model performed well" without a baseline
- cite plots without interpretation
- ignore data leakage or candidate-pool design
- treat graph centrality as automatically meaningful
- cannot reproduce commands
- cannot explain ethical access or personal-data risk

---

# Cross-Cutting Rubric: Interpretation, Discussion, and Depth

These expectations apply to every milestone and final deliverable.

## Interpretation Quality

Excellent interpretation:

- states what was learned
- states what was not learned
- explains why the evidence supports the claim
- connects the result to the product question
- identifies confounders, artifacts, and alternative explanations
- avoids overclaiming

Weak interpretation:

- describes a plot without explaining it
- reports a metric without context
- says a result is "good" without a baseline
- treats technical output as self-explanatory
- ignores uncertainty or limitations

## Discussion Quality

Excellent discussion:

- compares alternatives
- explains tradeoffs
- links choices across pipeline stages
- uses examples from the data
- acknowledges failure cases
- proposes realistic next steps

Weak discussion:

- summarizes steps chronologically without analysis
- repeats code outputs
- avoids explaining bad results
- gives generic future work

## Depth of Analysis

Excellent depth:

- includes parameter sweeps, ablations, baselines, or sensitivity checks
- shows failure analysis
- connects data quality to model behavior
- examines both aggregate metrics and concrete examples
- explains why results matter for the domain

Weak depth:

- uses a single default model run
- skips baselines
- has no error analysis
- has no examples
- treats outputs as final because the code ran

---

# Suggested Grading Workflow for Instructor

1. Check minimum completeness first.
2. Review milestone artifacts against the required deliverables.
3. Verify that the repository can be run from documented commands.
4. Read interpretation and limitation sections before assigning high marks.
5. During final defense, ask at least one question per project layer:
   - data
   - representation
   - clustering
   - recommendation/ranking
   - graph
   - reproducibility
   - ethics
6. Use the final defense score to reward genuine understanding and penalize shallow ownership.

---

# Student-Facing Summary

The strongest projects will not simply show many models.
They will show a coherent system.

To score highly, your team must demonstrate:

- a valid and ethical dataset
- a reproducible data pipeline
- meaningful feature representations
- rigorous clustering and validation
- a recommendation, ranking, or predictive decision engine with baselines
- a justified graph analysis
- clear interpretation and discussion
- honest limitations
- a working final demo
- a strong oral defense under technical questions

Each deliverable is graded out of `20`.
When the instructor combines deliverables into the final project grade, the final defense should carry the largest weight because it tests whether the team can explain and defend the project as engineers and analysts, not only submit artifacts.