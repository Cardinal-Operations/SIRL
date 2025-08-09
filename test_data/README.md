## Data descriptions
Our test_data mainly inherit from the OptMath projects, link, 

Our test data is primarily inherited from the OptMath project. You can find the original repository and a full description of the dataset at the following link: [OptMath-Github Repository](https://github.com/optsuite/OptMATH).

### Correction of the Mamo Dataset
#### Correction of TSP Instances 
We've adopted a three-steped approach to re-examine Traveling Salesman Problem (TSP) instances. 
 1. Extract TSP Instances: We first extract TSP instances from a predefined range. Specifically, we've focused on instances with indices from 59 to 98. Each instance is identified and described using a structured JSON format. For example:

```
    {"index": 50, "en_question": "Consider a courier company that needs to deliver packages to five distinct cities, denoted as E, F, G, H, and I. The courier can start from any city, but they must visit each city only once and then return to the starting point. The aim is to find a route that would minimize the total delivery cost. The cost might include factors like distance, fuel expenses, or traffic conditions. Here's an outline of the delivery cost between these cities:\nThe cost to deliver from City E to F is 50 units, to G is 48 units, to H is 99 units, and to I is 91 units.\nFrom City F, it costs 50 units to deliver to E, 57 units to deliver to G, 84 units to H, and 72 units to I.\nFor City G, the delivery costs are 48 units to E, 57 units to F, 46 units to H, and 86 units to I.\nIf the package starts from City H, it costs 99 units to deliver to E, 84 units to F, 46 units to G, and 29 units to I.\nLastly, from City I, it costs 91 units to deliver to E, 72 units to F, 86 units to G, and 29 units to H.\nWhat is the least total delivery cost for the courier to visit each city exactly once and then return to the starting point?"}
```


 2.  LLMs assistant including Deepseek-V3 and GPT-4o, are used to generate the cost matrix for each instance. This data is rigorously cross-validated by  human experts to ensure accuracy. A sample cost matrix is provided below:
    
    [[0, 50, 48, 99, 91],
    [50, 0, 57, 84, 72],
    [48, 57, 0, 46, 86],
    [99, 84, 46, 0, 29],
    [91, 72, 86, 29, 0]
    ]
    
   
3. Optimal Solution via Enumeration: The optimal solution for each instance is found using a brute-force enumeration method. The Python implementation below iterates through all possible permutations of the tour to identify the minimum-cost path
```python

import itertools

# number of cities
n = 5

# cost matrix
cost = [
    [0,  50, 48, 99, 91],  # E
    [50, 0,  57, 84, 72],  # F
    [48, 57, 0,  46, 86],  # G
    [99, 84, 46, 0,  29],  # H
    [91, 72, 86, 29, 0]    # I
]
# start city 0（City 1）
start_city = 0
other_cities = list(range(n))
other_cities.remove(start_city)

min_cost = float('inf')
best_tour = None

# employ an enumeration method that systematically explores all possible tours
for perm in itertools.permutations(other_cities):
    tour = [start_city] + list(perm) + [start_city]  # 
    cost_sum = sum(cost[tour[i]][tour[i+1]] for i in range(n))
    
    if cost_sum < min_cost:
        min_cost = cost_sum
        best_tour = tour

# the output 
print(f"The minimum cost: {min_cost}")
print("The optimal tour:", ' -> '.join(f"City {i+1}" for i in best_tour))

```


