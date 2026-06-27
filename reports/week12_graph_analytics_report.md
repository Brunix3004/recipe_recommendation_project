# Week 12 Graph Analytics and Centrality Report

## 1. Objective

This deliverable formalizes a domain graph for the Recipe Recommendation System and uses it for structural graph analysis. The Week 12 graph complements, but does not replace, the Week 10 recommender: the recommender asks what recipes a user may like, while this graph asks which ingredients are structurally central in the recipe corpus.

## 2. Formal Graph Definition

| Choice | Definition |
| --- | --- |
| Graph name | Ingredient-Ingredient Co-occurrence Graph |
| Graph type | undirected weighted graph |
| Nodes | A node is a normalized ingredient. |
| Node grain | One node represents one normalized ingredient token, not a recipe, user, or category. |
| Edges | An undirected edge connects two ingredients if they co-occur in at least one recipe. |
| Edge weight | The edge weight is the number of distinct recipes in which the two ingredients co-occur. |
| Directionality | The graph is undirected because ingredient co-occurrence has no natural direction. |
| Filtering thresholds | Nodes require recipe_count >= 50; edges require cooccurrence_count >= 20. |
| Recipe contribution rule | Each recipe contributes at most one count to an ingredient node and at most one count to a given ingredient pair. |

This graph is meaningful for the project because it captures ingredient relationship structure. It can reveal pantry staples, bridge ingredients, and ingredient neighborhoods that support ingredient exploration, substitution hypotheses, and future graph-aware recipe intelligence.

## 3. Graph Construction Pipeline

Input data came from `data/processed/recipes_processed.csv`. The pipeline parses `RecipeIngredientParts`, normalizes ingredient strings, deduplicates ingredients within each recipe, sorts them for deterministic pair generation, counts ingredient frequencies and pair co-occurrences, applies node and edge thresholds, then persists the graph as CSV, GraphML, JSON, figures, and this report.

Regeneration command:

```bash
python src/graphs/build_ingredient_graph.py \
  --recipes data/processed/recipes_processed.csv \
  --out artifacts/week12/ingredient_graph \
  --figures reports/figures \
  --report reports/week12_graph_analytics_report.md \
  --min-node-recipe-count 50 \
  --min-edge-recipe-count 20 \
  --sensitivity-edge-thresholds 5 10 20 50
```

## 4. Graph Summary and Validity Checks

| Metric | Value |
| --- | --- |
| Recipes processed | 522,517 |
| Recipes with >= 1 valid ingredient | 520,298 |
| Recipes with >= 2 valid ingredients | 511,496 |
| Raw unique ingredients | 7,287 |
| Retained ingredient nodes | 2,529 |
| Raw unique ingredient pairs | 974,020 |
| Retained edges | 88,757 |
| Density | 0.0278 |
| Sparsity | 0.9722 |
| Connected components | 84 |
| Largest component share | 0.9672 |
| Isolated nodes | 83 |
| Isolated node share | 0.0328 |
| Average degree | 70.1914 |
| Average weighted degree | 11243.9359 |

The graph is sparse, which is expected for a thresholded ingredient network. Most retained ingredients participate in a shared co-occurrence space. The selected edge threshold keeps repeated co-occurrence relationships while filtering one-off pair noise.

Validity checks:

| Check | Value | Status | Note |
| --- | --- | --- | --- |
| recipes_processed | 522,517 | ok | Number of processed recipe rows loaded from the input file. |
| recipes_with_at_least_1_valid_ingredient | 520,298 | ok | Rows with at least one parsed and normalized ingredient token. |
| recipes_with_at_least_2_valid_ingredients | 511,496 | ok | Rows able to contribute at least one ingredient pair. |
| raw_unique_ingredients_before_filtering | 7,287 | ok | Ingredient vocabulary before min-node filtering. |
| retained_ingredient_nodes_after_filtering | 2,529 | ok | Ingredient nodes with recipe_count above the selected threshold. |
| raw_unique_ingredient_pairs_before_filtering | 974,020 | ok | Unique unordered ingredient pairs before min-edge filtering. |
| retained_edges_after_filtering | 88,757 | ok | Ingredient pairs with co-occurrence counts above the selected threshold. |
| graph_density | 0.0278 | ok | A high density would indicate that the edge threshold is too permissive. |
| graph_sparsity | 0.9722 | ok | Graph sparsity is 1 - density. |
| isolated_node_count | 83 | ok | Retained nodes that have no retained edges after edge filtering. |
| isolated_node_share | 0.0328 | ok | Share of retained ingredient nodes that are isolated. |
| connected_components | 84 | ok | Fragmentation warning triggers if the largest component is below 60% of nodes. |
| largest_component_share | 0.9672 | ok | Share of retained nodes in the largest connected component. |
| generic_ingredient_dominance_top10_pagerank | 1 | warning | Warning triggers if at least 80% of top-10 PageRank ingredients are generic pantry staples. |

Figures:
- `reports/figures/ingredient_graph_degree_distribution.png`
- `reports/figures/ingredient_graph_weighted_degree_distribution.png`
- `reports/figures/ingredient_graph_component_size_distribution.png`

## 5. Connected Components

The graph contains 84 connected components. The largest component contains 2,446 ingredients, or 0.9672 of retained nodes.

| Component | Size | Share | Top by recipe count | Top by PageRank |
| --- | --- | --- | --- | --- |
| 1 | 2,446 | 0.9672 | salt; butter; sugar; onion; eggs; water; olive_oil; flour; milk; garlic_cloves | salt; butter; onion; sugar; olive_oil; water; garlic_cloves; eggs; flour; milk |
| 2 | 1 | 0.000395413 | 2_fat_cottage_cheese | 2_fat_cottage_cheese |
| 3 | 1 | 0.000395413 | absolut_citron_vodka | absolut_citron_vodka |
| 4 | 1 | 0.000395413 | beef_sirloin_steaks | beef_sirloin_steaks |
| 5 | 1 | 0.000395413 | beefsteak_tomato | beefsteak_tomato |
| 6 | 1 | 0.000395413 | best_foods_mayonnaise | best_foods_mayonnaise |
| 7 | 1 | 0.000395413 | black_currants | black_currants |
| 8 | 1 | 0.000395413 | blood_orange | blood_orange |
| 9 | 1 | 0.000395413 | boneless_pork_chop | boneless_pork_chop |
| 10 | 1 | 0.000395413 | bottled_water | bottled_water |

A large component means many ingredients participate in a shared culinary co-occurrence space. Smaller components may represent niche ingredients, rare cuisines, highly specialized recipe families, or noisy tokens. Isolated nodes are ingredients frequent enough to survive node filtering but without strong enough co-occurrence edges after the edge threshold.

## 6. Degree and Weighted Degree Analysis

Degree is the number of distinct ingredient neighbors. Weighted degree is the sum of co-occurrence counts across retained neighbors. A high degree ingredient appears with many different ingredients; a high weighted degree ingredient co-occurs frequently across the recipe corpus.

Top ingredients by weighted degree:

| Rank | Ingredient | Recipe count | Degree | Weighted degree | PageRank |
| --- | --- | --- | --- | --- | --- |
| 1 | salt | 190,464 | 2,222 | 1,610,641 | 0.0594 |
| 2 | butter | 123,598 | 1,657 | 917,003 | 0.0297 |
| 3 | sugar | 102,808 | 1,525 | 745,544 | 0.0258 |
| 4 | onion | 86,321 | 1,632 | 740,749 | 0.0265 |
| 5 | eggs | 80,436 | 1,251 | 621,655 | 0.0186 |
| 6 | olive_oil | 72,763 | 1,568 | 613,753 | 0.0223 |
| 7 | water | 79,884 | 1,736 | 609,510 | 0.0210 |
| 8 | garlic_cloves | 58,612 | 1,476 | 540,324 | 0.0189 |
| 9 | flour | 59,081 | 1,103 | 473,883 | 0.0135 |
| 10 | milk | 58,800 | 1,077 | 423,340 | 0.0129 |

Top ingredients by popularity baseline:

| Rank | Ingredient | Recipe count | Degree | Weighted degree | PageRank |
| --- | --- | --- | --- | --- | --- |
| 1 | salt | 190,464 | 2,222 | 1,610,641 | 0.0594 |
| 2 | butter | 123,598 | 1,657 | 917,003 | 0.0297 |
| 3 | sugar | 102,808 | 1,525 | 745,544 | 0.0258 |
| 4 | onion | 86,321 | 1,632 | 740,749 | 0.0265 |
| 5 | eggs | 80,436 | 1,251 | 621,655 | 0.0186 |
| 6 | water | 79,884 | 1,736 | 609,510 | 0.0210 |
| 7 | olive_oil | 72,763 | 1,568 | 613,753 | 0.0223 |
| 8 | flour | 59,081 | 1,103 | 473,883 | 0.0135 |
| 9 | milk | 58,800 | 1,077 | 423,340 | 0.0129 |
| 10 | garlic_cloves | 58,612 | 1,476 | 540,324 | 0.0189 |

Figure: `reports/figures/ingredient_graph_top_weighted_degree.png`

## 7. PageRank / Centrality Analysis

In this undirected weighted graph, PageRank models a random walk over the ingredient network. Ingredients connected to other important ingredients receive higher scores, and weighted PageRank uses co-occurrence strength as the transition weight.

Top ingredients by weighted PageRank:

| Rank | Ingredient | Recipe count | Degree | Weighted degree | PageRank |
| --- | --- | --- | --- | --- | --- |
| 1 | salt | 190,464 | 2,222 | 1,610,641 | 0.0594 |
| 2 | butter | 123,598 | 1,657 | 917,003 | 0.0297 |
| 3 | onion | 86,321 | 1,632 | 740,749 | 0.0265 |
| 4 | sugar | 102,808 | 1,525 | 745,544 | 0.0258 |
| 5 | olive_oil | 72,763 | 1,568 | 613,753 | 0.0223 |
| 6 | water | 79,884 | 1,736 | 609,510 | 0.0210 |
| 7 | garlic_cloves | 58,612 | 1,476 | 540,324 | 0.0189 |
| 8 | eggs | 80,436 | 1,251 | 621,655 | 0.0186 |
| 9 | flour | 59,081 | 1,103 | 473,883 | 0.0135 |
| 10 | milk | 58,800 | 1,077 | 423,340 | 0.0129 |
| 11 | pepper | 48,754 | 1,195 | 405,360 | 0.0124 |
| 12 | all_purpose_flour | 41,176 | 936 | 355,352 | 0.0101 |
| 13 | baking_powder | 39,354 | 642 | 343,248 | 0.0097 |
| 14 | brown_sugar | 40,810 | 912 | 323,737 | 0.0094 |
| 15 | garlic | 34,901 | 1,141 | 300,410 | 0.0093 |

High PageRank should be interpreted as structural centrality, not as "best ingredient." Generic staples may rank highly because they connect many parts of the corpus. More specific bridge ingredients are useful because they can connect ingredient neighborhoods that would otherwise be less directly linked.

Figures:
- `reports/figures/ingredient_graph_top_pagerank.png`
- `reports/figures/ingredient_graph_pagerank_vs_popularity.png`

## 8. Comparison: Graph Ranking vs Popularity Baseline

The popularity baseline is `recipe_count`: the number of distinct recipes containing an ingredient. The graph rankings are weighted PageRank and weighted degree.

Model-based recipe ranking artifacts were not available, so the comparison uses ingredient recipe frequency as the popularity baseline.

Spearman correlations:

| Baseline metric | Graph metric | Spearman rho |
| --- | --- | --- |
| recipe_count | weighted_degree | 0.9754 |
| recipe_count | pagerank_weighted | 0.9762 |

Top-K overlap with popularity:

| Graph metric | K | Overlap count | Overlap share |
| --- | --- | --- | --- |
| pagerank_weighted | 10 | 10 | 1 |
| pagerank_weighted | 20 | 19 | 0.9500 |
| pagerank_weighted | 50 | 48 | 0.9600 |
| pagerank_weighted | 100 | 95 | 0.9500 |
| weighted_degree | 10 | 10 | 1 |
| weighted_degree | 20 | 19 | 0.9500 |
| weighted_degree | 50 | 48 | 0.9600 |
| weighted_degree | 100 | 95 | 0.9500 |

Examples where PageRank ranks ingredients higher than raw frequency:

| Ingredient | Recipe count | Popularity rank | PageRank rank | PageRank | Rank gain |
| --- | --- | --- | --- | --- | --- |
| yellow_zucchini | 70 | 2,220 | 1,357 | 7.69234e-05 | 863 |
| green_zucchini | 106 | 1,869 | 1,195 | 8.57968e-05 | 674 |
| regular_margarine | 50 | 2,526 | 1,866 | 6.56802e-05 | 660 |
| black_cardamom_pods | 102 | 1,909 | 1,275 | 8.15695e-05 | 634 |
| cake_yeast | 63 | 2,307 | 1,688 | 6.82883e-05 | 619 |

Examples where popularity is high but PageRank is less distinctive:

| Ingredient | Recipe count | Popularity rank | PageRank rank | PageRank | Rank loss |
| --- | --- | --- | --- | --- | --- |
| sweet_vermouth | 124 | 1,728 | 2,518 | 6.10141e-05 | 790 |
| cream_cheese_spread | 136 | 1,654 | 2,440 | 6.1584e-05 | 786 |
| amarula_cream_liqueur | 137 | 1,643 | 2,387 | 6.17136e-05 | 744 |
| seltzer_water | 232 | 1,285 | 1,959 | 6.47025e-05 | 674 |
| vanilla_vodka | 323 | 1,062 | 1,734 | 6.74841e-05 | 672 |

If the correlations are high, graph centrality is partially driven by ingredient popularity. Differences between the rankings show where PageRank captures network position beyond simple frequency.

## 9. Sensitivity Analysis

Edge threshold matters because it controls whether weak one-off co-occurrences are retained. Lower thresholds keep more edges and usually create a denser, more connected graph. Higher thresholds emphasize stable co-occurrences but can isolate nodes and fragment components.

| Edge threshold | Nodes | Edges | Density | Sparsity | Components | Largest share | Isolates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5 | 2,529 | 249,334 | 0.0780 | 0.9220 | 1 | 1 | 0.0000 |
| 10 | 2,529 | 151,435 | 0.0474 | 0.9526 | 3 | 0.9992 | 2 |
| 20 | 2,529 | 88,757 | 0.0278 | 0.9722 | 84 | 0.9672 | 83 |
| 50 | 2,529 | 41,535 | 0.0130 | 0.9870 | 772 | 0.6951 | 771 |

Top-20 PageRank overlap between thresholds:

| Threshold A | Threshold B | Overlap count | Jaccard overlap |
| --- | --- | --- | --- |
| 5 | 10 | 20 | 1 |
| 5 | 20 | 20 | 1 |
| 5 | 50 | 20 | 1 |
| 10 | 20 | 20 | 1 |
| 10 | 50 | 20 | 1 |
| 20 | 50 | 20 | 1 |

Stable top central ingredients indicate robust graph structure. Large overlap changes indicate sensitivity to the edge-definition threshold. Figure: `reports/figures/ingredient_graph_sensitivity_edges.png`

## 10. Interpretation Note: What the Graph Means and Does Not Mean

Graph structure means ingredient co-occurrence patterns in the Food.com dataset. It can reflect culinary compatibility inside the dataset, pantry-staple centrality, bridge ingredients between culinary styles, and structural ingredient importance.

Graph structure does not mean user preference, nutritional quality, causal compatibility, substitution equivalence, personalized recommendation, or universal cultural importance. It also does not prove that two ingredients taste good together outside the dataset context. Food.com may overrepresent American and Western comfort food, so central ingredients reflect the dataset's cuisine distribution and contributor behavior.

## 11. Relationship to Previous Deliverables

Week 5 produced content embeddings from ingredient, category, keyword, and numeric features. Week 7 clustered recipes using semantic and numeric recipe representations. Week 10 ranked recipes using user behavior and content similarity. Week 12 adds a structural ingredient-network perspective that can support future cluster-aware and graph-aware recommendation.

## 12. Limitations and Future Work

Ingredient normalization may not merge all synonyms. Raw co-occurrence favors common ingredients. The edge threshold affects graph density and component structure. Quantities, preparation instructions, and cooking order are ignored. The graph is undirected and does not capture preparation sequence. High centrality may be dominated by staples.

Future work could use PMI, PPMI, or Jaccard-weighted graphs; recipe-recipe graphs; user-recipe bipartite graphs; synonym dictionaries; or graph-aware recommendation features.

## 13. Reproducibility

Exact command:

```bash
python src/graphs/build_ingredient_graph.py \
  --recipes data/processed/recipes_processed.csv \
  --out artifacts/week12/ingredient_graph \
  --figures reports/figures \
  --report reports/week12_graph_analytics_report.md \
  --min-node-recipe-count 50 \
  --min-edge-recipe-count 20 \
  --sensitivity-edge-thresholds 5 10 20 50
```

Generated artifacts under `artifacts/week12/ingredient_graph`:

- `artifacts/week12/ingredient_graph/ingredient_nodes.csv`
- `artifacts/week12/ingredient_graph/ingredient_edges.csv`
- `artifacts/week12/ingredient_graph/ingredient_graph.graphml`
- `artifacts/week12/ingredient_graph/ingredient_graph_summary.json`
- `artifacts/week12/ingredient_graph/connected_components.csv`
- `artifacts/week12/ingredient_graph/graph_validity_checks.csv`
- `artifacts/week12/ingredient_graph/comparison_graph_vs_popularity.csv`
- `artifacts/week12/ingredient_graph/top_ingredients_by_popularity.csv`
- `artifacts/week12/ingredient_graph/top_ingredients_by_weighted_degree.csv`
- `artifacts/week12/ingredient_graph/top_ingredients_by_pagerank.csv`
- `artifacts/week12/ingredient_graph/top_ingredients_by_log_pagerank.csv`
- `artifacts/week12/ingredient_graph/sensitivity_edge_thresholds.csv`
- `artifacts/week12/ingredient_graph/sensitivity_top20_pagerank_overlap.csv`
- `artifacts/week12/ingredient_graph/graph_pipeline_config.json`

Generated figures:

- `reports/figures/ingredient_graph_degree_distribution.png`
- `reports/figures/ingredient_graph_weighted_degree_distribution.png`
- `reports/figures/ingredient_graph_component_size_distribution.png`
- `reports/figures/ingredient_graph_pagerank_vs_popularity.png`
- `reports/figures/ingredient_graph_top_pagerank.png`
- `reports/figures/ingredient_graph_top_weighted_degree.png`
- `reports/figures/ingredient_graph_sensitivity_edges.png`
