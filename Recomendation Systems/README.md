# Recommendation Systems

Recommendation Systems are an integral part of our lives. We see them everywhere is our everyday's life: Amazon products recommendations, Youtube video recommendation, Spotify music, etc.

In this section of my GitHub I would like to share information that I have learn't about Rec. Systems: Machine Learning methods, Artificial Intelligence methods and so on.

## Losses

Loss Functions for recommendation systems:

### Hit Rate

Hit rate in the context of recommendation systems is a simple yet insightful metric used to evaluate the quality of recommendations. It indicates how often the items recommended by the system are relevant or of interest to the user. Broadly, the hit rate can be defined as the proportion of instances where at least one recommended item was successfully adopted by the user.

Within recommendation systems, a "hit" is defined as an event where a recommended item (e.g., product, movie, book) matches the user's interests, and they engage with that item through a target action. This action varies depending on the type of recommendation system and task, such as purchasing a product, watching a movie, or reading an article.

Hit rate is typically calculated as the ratio of the number of "hits" to the total number of recommendations presented to users, or as the proportion of user sessions in which at least one "hit" occurred. This can be expressed by the formula: `hit rate = number of sessions with at least one hit / total number of sessions`

### Average Reciprocal hit rate (ARHR)

Based on the position of the predicted item we estimate how much score the prediction will get: `sum(1/rank_i) / users`

`1/rank_i` is reciprocal rank which means that if the `rank_i` is too far away in the prediction list, we then receive smaller rank from this item, hence the overall loss becomes higher.

### Cumulative hit rate (cHR)

Throw away items with bad user ratings. We don't take into account items which are underrated by users thus let our system to learn only on high-rated items.

### Other Metrics

- **Similarity** - how items we recommend are similar to each other.
- **Diversity** - how different the items are. This is not always a good score though because user can start getting a lot of random recommendations they don't want. Can be computed like: `1 - Similarity`
- **Coverage** - how many items can our system cover while making predictions
- **Novelty** - how popular the items are that we are recommending
- **Churn** - how often do recommendations change
- **Responsiveness** - how quickly does new user behavior influence your recommendations
- **A/B tests** - online testing on users
