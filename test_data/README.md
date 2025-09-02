# Benchmark Data Descriptions

Our test data is primarily inherited from the OptMath project. 
You can find the original repository and a full description of the dataset at the following link: [OptMath-Github Repository](https://github.com/optsuite/OptMATH).
 We performed a quick review and correction on the NL4OPT, IndustryOR, MAMO-ComplexLP, and MAMO-EasyLP datasetsets. 
 Table 1 summarizes the sample counts for each dataset before and after our revisions.  
|Dataset|	Number Before Correction	| Number After Correction |
| ----- | ---------------- | -------------- |
| NL4OPT   | 245           | 245     |
| MAMO_EasyLP   |    652          |  642      |
| MAMO_ComplexLP   | 211                | 203             |
| IndustryOR   | 100               | 100            |
| OptMATH   | 166            | 166        |
| Optibench   | 605              | 605            |

In the subsequent sections, we detail the specific corrections made to the MAMO_EasyLP and MAMO_ComplexLP datasets.

## Correction of the MAMO_EasyLP Dataset.
Here is a overview of the correction of the MAMO_EasyLP dataset

| Type of Correction |   Problem index list |
| ----- | ---------------- |
| Removed   | 40 , 43 , 150 , 157 , 204 , 238 , 265 , 268 , 562 , 650    |
| Question Revision   | 172 , 178 , 236 , 364 , 405 , 414 , 505 , 613    |
| Answer Correction   | 2 , 34 , 35 , 71 , 98 , 99 , 111 , 114 , 129 , 131 , 149 , 155 , 160 , 162 , 166 , 172 , 173 , 178 , 203 , 222 , 250 , 252 , 254 , 257 , 266 , 267 , 273 , 275 , 289 , 307 , 313 , 321 , 322 , 323 , 335 , 345 , 358 , 364 , 392 , 398 , 414 , 434 , 435 , 436 , 453 , 471 , 479 , 480 , 485 , 505 , 514 , 550 , 556 , 561 , 571 , 582 , 585 , 598 , 604 , 605 , 606 , 608 , 613 , 615 , 621 , 630 , 636 , 638 , 642 , 648    |

### Type of Correction: Removed
These questions were removed from the dataset due to incomplete conditions, contradictory requirements, or the fact that they admit only trivial solutions. This section highlights the types of flaws identified and provides specific examples to illustrate our rationale.

------
Question 204 was removed because it contained contradictory conditions:
```
    {"index": 204,"en_question": "In a financial planning scenario, an investor is considering allocating funds across three different investment options: $X$, $Y$, and $Z$. These options could represent stocks, bonds, or real estate investments. The total funds available for all three investments combined cannot exceed $\\$1000$ due to budget constraints. Investment option $X$ requires a minimum investment of $\\$300$ to be viable, while investment option $Y$ has a maximum allowable investment of $\\$500$ due to risk management considerations. Investment option $Z$, on the other hand, requires a minimum investment of $\\$200$, possibly due to its larger scale or higher initial setup costs. Each unit of investment in options $X$, $Y$, and $Z$ yields different returns or costs, quantified as $\\$10$, $\\$15$, and $\\$20$, respectively. The investor aims to minimize the total cost while adhering to these constraints (the number of units for each type of investments should be integers). Calculate the minimum total cost required for this scenario in dollars.", "en_answer": "7000"}
```
This problem was removed because of an inherent logical flaw. It requests the minimization of "total cost" but then states that the per-unit values represent "returns or costs." This ambiguity creates a contradiction that makes the problem unsolvable as stated.

------

Question 221 is flawed because it lacks crucial data:



```
    {"index": 221,"en_question": "A retail manager is planning the distribution of budget among three departments: X (Buying), Y (Marketing), and Z (Logistics). The total budget for all departments combined is limited to a maximum of 1000 units due to financial constraints. Department X requires at least 200 units, department Y needs a minimum of 150 units, and department Z must receive at least 50 units to function effectively. Each unit invested in departments X, Y, and Z generates returns quantified as 2, 3, and 1 unit respectively. The retail manager aims to minimize the total cost while meeting these investment constraints for each department. Given that all investments need to be whole numbers due to accounting policies, calculate the minimum total cost required for this scenario.", "en_answer": "900"}
```

In this case, the question lacks information about the costs for departments X, Y, and Z. Because of this missing data, LLMs cannot generate an exact answer.

------

Question 157 has been removed since it becomes invalid when all decision variables are equal to zero：

```
    {“index”: 157, "en_question": "A real estate investor is planning to invest in two types of properties, $X$ and $Y$. The investor cannot purchase more than a total of 1000 properties due to financial constraints. To ensure diversity in the investment portfolio, the number of type $X$ properties purchased should be at least three times the number of type $Y$ properties sold. Each property of type $X$ costs $\\$300,000$, and each property of type $Y$ costs $\\$200,000$. Given these conditions and aiming for whole numbers of properties due to the nature of real estate transactions, what is the minimum total cost for purchasing these properties? Provide your answer rounded to the nearest dollar.", "en_answer": "50100000"}
```

It is evident that this question admits a trivial solution at (0,0), which indicates that the problem is not well-posed.

### Type of Correction: Question Revision

Although certain questions exhibit internal inconsistencies, their overall formulation remains coherent, making it possible to revise them into valid and meaningful problems.

------
Question 364:

```
    {"index": 364,"en_question": "A farmer is planning to plant four different crops: Wheat, Corn, Soybean, and Rice in his field. Due to various factors like the suitability of the crop to the local climate and soil type, the farmer's experience with growing each crop, market prices, etc., the net profit per acre for each crop is different: \\$4 for Wheat, \\$3 for Corn, \\$2 for Soybean, and \\$1 for Rice.\\n\\nThe total area available for planting Wheat and Corn combined cannot exceed 1000 acres due to certain restrictions on these types of crops. Similarly, the total area that can be used to grow Soybean and Rice together is limited to 800 acres due to water availability constraints.\\n\\nTo maintain a diverse crop portfolio and reduce risk from price fluctuations or disease outbreaks affecting one type of crop more than others, the farmer wants at least 200 more acres of Wheat than Soybean. Also considering marketing strategy and demand-supply balance in local markets, he doesn't want more than 500 acres difference between areas planted with Corn and Rice.\\n\\nFurthermore,- The maximum area that can be allocated to Wheat is 600 acres.- For Corn it's capped at 400 acres.- Soybean can be grown on up-to 350 acres.- While there are no lower bounds on how much area can be used for any crop (i.e., it could even be zero), but he can't plant more than 300 acres of Rice due to its high water requirement.\\n\\nGiven these conditions along with the fact that all acreages must be whole numbers as you cannot practically have non-integral amounts of land being cultivated. What should be his planting plan if he wants to maximize his profits? Provide your answer as minimum total cost rounded off to nearest dollar.", "en_answer": "800"}

```

In this case, the two statements regarding impact points are contradictory:

- What should be his planting plan if he wants to maximize his profits?
- Provide your answer as minimum total cost rounded off to nearest dollar.

We rivised the second statement into "Provide your answer as maximum total profits rounded off to nearest dollar" and verified the answer.


### Type of Correction: Correcting Answer
We performed a meticulous review of the benchmark questions with the assistance of several optimization experts. This review identified inaccuracies in the provided answers, which we have since verified and corrected. The updated answers are presented below.

-------


| Index | Original Answer | Correct Answer| 
| --- | --- | --- |
| 2 | 168 | 20 |
| 34 | 340 | 590 |
| 35 | 102000 | 25000 | 
| 71 | 1340 | 3000 |
| 98 | 20000 | 5000 |
| 99 | 20000 | 400000|
| 111 | 30 | 52 |
| 114 | 15000 | 15600 |
| 129 | 210 | 400 |
| 131 | 200000 | 4000 |
| 149 | 750 | 800 |
| 155 | 500000 | 1700000 |
| 160 | 250000 | 500000 |
| 162 | 500000 | 900000 |
| 166 | 4000 | 4670 |
| 172 | 50000 | 181 |
| 173 | 40020 | 49420 |
| 178 | 38000 | 130000 |
| 203 | 18500 | 26500 |
| 222 | 1650 | 4400 |
| 250 | 15 | 7 |
| 252 | 1000 | 1300 |
| 254 | 800 | 2000 |
| 257 | 50000 | 86680 |
| 266 | 2500 | 4150 |
| 267 | 49000 | 89000 |
| 273 | 10 | 15 |
| 275 | 60 | 120 |
| 289 | 150 | 210 |
| 307 | 2000 | 4000 |
| 313 | 1400 | 1000 |
| 321 | 74 | 70 |
| 322 | 90 | 110 |
| 323 | 150000 | 350000 |
| 335 | 60000 | 60500 |
| 345 | 601 | 566.67 |
| 358 | 8000 | 9000 |
| 364 | 800 | 4600 |
| 392 | 380 | 300 |
| 398 | 1760 | 1740 |
| 414 | 500000 | 5 |
| 434 | 100000 | 3335 |
| 435 | 9000 | 11500 |
| 436 | 18000 | 360 |
| 453 | 2500 | 4450 |
| 471 | 120 | 140 |
| 479 | 28000 | 25000 |
| 480 | 50000 | 55000 |
| 485 | 150 | 262 |
| 505 | 25000 | 500 |
| 514 | 200 | 218 |
| 550 | 370000 | 570000 |
| 556 | 20 | 16 |
| 561 | 130200 | 31200 |
| 571 | 85000 | 250 |
| 582 | 340000 | 2400 |
| 585 | 420 | 0.66 |
| 598 | 6600 | 7350 |
| 604 | 13000 | 9000 |
| 605 | 1684 | 2160 |
| 606 | 2550 | 2000 |
| 608 | 57000 | 56000 |
| 613 | 27 | 22 |
| 615 | 260000 | 300000 |
| 621 | 2267 | 600 |
| 630 | 22000 | 21000 |
| 636 | 20000 | 56500 |
| 638 | 2000000 | 24100 |
| 642 | 3333.333 | -9999 |
| 648 | 2000000 | 4288000 |


## Correction of the MAMO_ComplexLP Dataset

 
Here is a summary of the corrections made to the MAMO_ComplexLP dataset. This process included revisions to Traveling Salesman Problem (TSP) instances, Capacitated Facility Location Problem (CFLP) instances, and other suspicious instances that required closer inspection.

| Type of Correction |   Problem index list |
| ----- | ---------------- |
| Removed   | 145 , 146 , 148 , 149 , 150 , 159 , 170 , 172    |
| Question Revision   | 163 , 169    |
| Answer Correction   | 59 , 60 , 61 , 62 , 64 , 65 , 66 , 68 , 69 , 70 , 71 , 72 , 74 , 75 , 76 , 77 , 78 , 79 , 80 , 81 , 82 , 84 , 85 , 87 , 88 , 89 , 90 , 91 , 92 , 93 , 94 , 95 , 96 , 97 , 98 , 177 , 183 , 185 , 186 , 196 , 198 , 199 , 200  |

### Type of Correction: Removed
Due to their incomplete conditions, a number of questions related to the Capacitated Facility Location Problem (CFLP) were removed from the dataset. This section outlines the various types of flaws identified and presents specific examples to illustrate the rationale behind their exclusion.

------

Question 145:

```
    {"index": 145, "en_question": "In the bustling landscape of retail and distribution, LogicFlow Inc. is at a pivotal juncture aiming to streamline its supply chain for maximum efficiency and minimal cost. The company needs to determine the best strategy for distributing products to its eight retail stores, which are scattered across different locations. The challenge lies in selecting from eight potential distribution centers, each with its unique opening cost and varying transportation costs per unit to the stores. The goal is clear: to fulfill the demand of every retail store while keeping the overall costs as low as possible.\n\nThe specifics of this optimization scenario are as follows:\n\n- There are 8 potential Distribution Centers that LogicFlow Inc. is considering.\n- The company operates 8 Retail Stores requiring supplies.\n\nThe Opening Costs for Each Distribution Center (in dollars) are listed below:\n- Distribution Center 1: $151,000\n- Distribution Center 2: $192,000\n- Distribution Center 3: $114,000\n- Distribution Center 4: $171,000\n- Distribution Center 5: $160,000\n- Distribution Center 6: $120,000\n- Distribution Center 7: $182,000\n- Distribution Center 8: $186,000\n\nTransportation Costs Per Unit from Each Distribution Center to Each Retail Store (in dollars) are detailed as follows:\n- From Distribution Center 1 to Stores: $3 to Store 1, and so on, up to $4 to Store 8.\n- From Distribution Center 2 to Stores: $2 to Store 1, up to $4 to Store 8.\n- This pattern continues, reflecting the variability and strategic considerations in choosing the optimal distribution center locations and the dispatch of goods to the retail stores.\n\nThe Demand of Each Retail Store (in units) is:\n- Store 1: 695 units, continuing in a similar manner up to Store 8: 440 units.\n\nThe Supply Capacity of Each Distribution Center (in units) is as follows:\n- Distribution Center 1: 1980 units, with similar details provided for each center, ending with Distribution Center 8: 1962 units.\n\nGiven these particulars, LogicFlow Inc. must make a calculated decision on which distribution centers to open and the most cost-efficient way to transport goods to meet the demands of all eight retail stores. The decision matrix includes opening costs for each center, the transportation cost per unit to each store, and the capacity to meet the store demands.\n\nQuestion:\nWhat is the optimal total cost for LogicFlow Inc. to open the necessary distribution centers and transport goods in order to satisfy the demands of all eight retail stores? This total cost must include both the opening costs of the centers and the transportation costs of supplying the stores.", "en_answer": "443177.0"}
```

In this case, the data is fundamentally **incomplete**. The flaw manifests in two key areas:

   *  **Transportation Costs**: The cost matrix is only partially described, lacking explicit values for all distribution center-store pairs.
   *  **Demand and Supply Capacities**: The problem's constraints are ambiguously defined, with details such as "continuing in a similar manner" or "with similar details provided."


### Type of Correction: Question Revision

This section details the revised versions of CFLP questions 163 and 169, which were corrected by incorporating the originally missing conditions.

------

The original questions are as follows.

```
    {"index": 163, "en_question": "In the bustling marketplace of today, the fictional company GlobalSupply Inc. stands at a crucial juncture. It seeks to efficiently bridge its supply chain, connecting six potential distribution centers with nine eager retail stores scattered across the region. The objective is clear yet challenging: to minimize the total cost involved in opening distribution centers and transporting goods to fulfill the demands of the retail stores. This total cost encapsulates both the opening expenses of the distribution centers and the variable transportation costs per unit of goods delivered to the stores.\n\nHere are the specifics:\n\n- There are six Distribution Centers (DCs) under consideration for opening.\n- Nine Retail Stores are in need of regular supplies.\n\nThe Opening Costs for each Distribution Center (in dollars) are as follows:\n- DC 1: $151,000\n- DC 2: $192,000\n- DC 3: $114,000\n- DC 4: $171,000\n- DC 5: $160,000\n- DC 6: $120,000\n\nTransportation Cost Per Unit from each Distribution Center to each Retail Store (in dollars):\n- From DC 1 to Stores: $2 to Store 1, $3 to Store 2, and so on, up to $2 to Store 9\n- From DC 2 to Stores: $4 to Store 1, $2 to Store 2, and so on, up to $4 to Store 9\n- From DC 3 to Stores: $1 to Store 1, $1 to Store 2, and so on, up to $4 to Store 9\n- From DC 4 to Stores: $4 to Store 1, $1 to Store 2, and so on, up to $4 to Store 9\n- From DC 5 to Stores: $1 to Store 1, $4 to Store 2, and so on, up to $4 to Store 9\n- From DC 6 to Stores: $4 to Store 1, $4 to Store 2, and so on, up to $4 to Store 9\n\nThe Demand of each Retail Store (in units) ranges from 401 units for Store 1 to 955 units for Store 7, with varying demands for the others in between.\n\nThe Supply Capacity of each Distribution Center (in units) is also diverse, with DC 1 capable of supplying 1795 units, down to DC 6 which can supply 1095 units, and various capacities for the others.\n\nGlobalSupply Inc. now faces the daunting task of determining the optimal combination of distribution centers to open and the distribution of supplies to the retail stores in a way that the total opening and transportation costs are minimized.\n\n**Question:**\nWhat is the optimal total cost for GlobalSupply Inc. to both open the necessary distribution centers and transport goods to meet the demands of all nine retail stores, considering the given capacities and costs?", "en_answer": "607479.0"}
```
```
    {"index": 169, "en_question": "In the bustling world of retail, GlobalSupply Inc. is at a crossroads, needing to efficiently distribute its products to eight retail stores scattered across the region. To achieve this, GlobalSupply Inc. must select from six potential distribution centers, each with its unique set of opening costs and capacities. The challenge lies not only in selecting which distribution centers to open but also in determining how to distribute their products in a way that minimizes the total cost, including both the opening of distribution centers and the transportation of goods to the stores.\n\nGiven Data:\n\n- Number of Potential Distribution Centers: 6\n- Number of Retail Stores Needing Supply: 8\n\nOpening Costs for Each Distribution Center (in dollars):\n- Center 1: $151,000\n- Center 2: $192,000\n- Center 3: $114,000\n- Center 4: $171,000\n- Center 5: $160,000\n- Center 6: $120,000\n\nTransportation Cost Per Unit from Each Distribution Center to Retail Stores (in dollars):\n- From Center 1: $2 to Store 1, $3 to Store 2, and so on up to $5 to Store 8\n- From Center 2: $2 to Store 1, $4 to Store 2, and so on up to $2 to Store 8\n- From Center 3: $5 to Store 1, $4 to Store 2, and so on up to $4 to Store 8\n- From Center 4: $4 to Store 1, $3 to Store 2, and so on up to $3 to Store 8\n- From Center 5: $5 to Store 1, $1 to Store 2, and so on up to $2 to Store 8\n- From Center 6: $1 to Store 1, $2 to Store 2, and so on up to $4 to Store 8\n\nDemand of Each Retail Store (in units):\n- Store 1: 908 units\n- Store 2: 434 units\n- Store 3: 605 units\n- Store 4: 480 units\n- Store 5: 961 units\n- Store 6: 787 units\n- Store 7: 401 units\n- Store 8: 789 units\n\nSupply Capacity of Each Distribution Center (in units):\n- Center 1: 1365 units\n- Center 2: 1929 units\n- Center 3: 1502 units\n- Center 4: 1201 units\n- Center 5: 1529 units\n- Center 6: 961 units\n\nThe optimal solution must navigate the intricate balance between opening costs, transportation costs, supply capacities, and the demands of the retail stores. \n\nQuestion:\nWhat is the optimal total cost for GlobalSupply Inc. to open distribution centers and transport goods to meet all the retail stores' demands, considering the cost of opening the centers and the transportation costs of supplying the stores?", "en_answer": "587428.0"}
```
------
The revised questions are as follows.
```
    {"index": 163, "en_question": "In the bustling marketplace of today, the fictional company GlobalSupply Inc. stands at a crucial juncture. It seeks to efficiently bridge its supply chain, connecting six potential distribution centers with nine eager retail stores scattered across the region. The objective is clear yet challenging: to minimize the total cost involved in opening distribution centers and transporting goods to fulfill the demands of the retail stores. This total cost encapsulates both the opening expenses of the distribution centers and the variable transportation costs per unit of goods delivered to the stores.\n\nHere are the specifics:\n\n- There are six Distribution Centers (DCs) under consideration for opening.\n- Nine Retail Stores are in need of regular supplies.\n\nThe Opening Costs for each Distribution Center (in dollars) are as follows:\n- DC 1: $151,000\n- DC 2: $192,000\n- DC 3: $114,000\n- DC 4: $171,000\n- DC 5: $160,000\n- DC 6: $120,000\n\nTransportation Cost Per Unit from each Distribution Center to each Retail Store (in dollars):\n- From DC 1 to Stores: $2 to Store 1, $3 to Store 2, $2 to Store 3, $5 to Store 4, $1 to Store 5, $3 to Store 6, $4 to Store 7, $2 to Store 8, $1 to Store 9.\n- From DC 2 to Stores: $4 to Store 1, $2 to Store 2, $4 to Store 3, $2 to Store 4, $3 to Store 5, $1 to Store 6, $3 to Store 7, $5 to Store 8, $3 to Store 9.\n- From DC 3 to Stores: $1 to Store 1, $1 to Store 2, $1 to Store 3, $3 to Store 4, $5 to Store 5, $4 to Store 6, $5 to Store 7, $1 to Store 8, $4 to Store 9.\n- From DC 4 to Stores: $4 to Store 1, $1 to Store 2, $5 to Store 3, $5 to Store 4, $2 to Store 5, $2 to Store 6, $1 to Store 7, $4 to Store 8, $2 to Store 9.\n- From DC 5 to Stores: $1 to Store 1, $4 to Store 2, $2 to Store 3, $1 to Store 4, $3 to Store 5, $3 to Store 6, $2 to Store 7, $3 to Store 8, $3 to Store 9.\n- From DC 6 to Stores: $4 to Store 1, $4 to Store 2, $3 to Store 3, $2 to Store 4, $1 to Store 5, $1 to Store 6, $4 to Store 7, $1 to Store 8, $5 to Store 9.\n\nDemand of Each Retail Store (in units):\n- Store 1: 401\n- Store 2: 524\n- Store 3: 901\n- Store 4: 626\n- Store 5: 729\n- Store 6: 420\n- Store 7: 955\n- Store 8: 935\n- Store 9: 985\n\nSupply Capacity of Each Distribution Center (in units):\n- DC 1: 1795\n- DC 2: 1400\n- DC 3: 1392\n- DC 4: 1625\n- DC 5: 1224\n- DC 6: 1095\nGlobalSupply Inc. now faces the daunting task of determining the optimal combination of distribution centers to open and the distribution of supplies to the retail stores in a way that the total opening and transportation costs are minimized.\n\n**Question:**\nWhat is the optimal total cost for GlobalSupply Inc. to both open the necessary distribution centers and transport goods to meet the demands of all nine retail stores, considering the given capacities and costs?\n", "en_answer": "722476.0"}
```
```
    {"index": 169, "en_question": "In the bustling world of retail, GlobalSupply Inc. is at a crossroads, needing to efficiently distribute its products to eight retail stores scattered across the region. To achieve this, GlobalSupply Inc. must select from six potential distribution centers, each with its unique set of opening costs and capacities. The challenge lies not only in selecting which distribution centers to open but also in determining how to distribute their products in a way that minimizes the total cost, including both the opening of distribution centers and the transportation of goods to the stores.\n\nGiven Data:\n\n- Number of Potential Distribution Centers: 6\n- Number of Retail Stores Needing Supply: 8\n\nOpening Costs for Each Distribution Center (in dollars):\n- Center 1: $151,000\n- Center 2: $192,000\n- Center 3: $114,000\n- Center 4: $171,000\n- Center 5: $160,000\n- Center 6: $120,000\n\nTransportation Cost Per Unit from Each Distribution Center to Retail Stores (in dollars):\n- From Center 1: $2 to Store 1, $3 to Store 2, $1 to Store 3, $3 to Store 4, $2 to Store 5, $4 to Store 6, $3 to Store 7, $5 to Store 8.\n- From Center 2: $2 to Store 1, $4 to Store 2, $4 to Store 3, $5 to Store 4, $5 to Store 5, $1 to Store 6, $2 to Store 7, $1 to Store 8.\n- From Center 3: $5 to Store 1, $4 to Store 2, $3 to Store 3, $2 to Store 4, $2 to Store 5, $1 to Store 6, $5 to Store 7, $5 to Store 8.\n- From Center 4: $4 to Store 1, $3 to Store 2, $5 to Store 3, $3 to Store 4, $3 to Store 5, $5 to Store 6, $1 to Store 7, $3 to Store 8.\n- From Center 5: $5 to Store 1, $1 to Store 2, $3 to Store 3, $5 to Store 4, $4 to Store 5, $4 to Store 6, $4 to Store 7, $2 to Store 8.\n- From Center 6: $1 to Store 1, $2 to Store 2, $1 to Store 3, $1 to Store 4, $2 to Store 5, $3 to Store 6, $3 to Store 7, $4 to Store 8.\n\nDemand of Each Retail Store (in units):\n- Store 1: 908 units\n- Store 2: 434 units\n- Store 3: 605 units\n- Store 4: 480 units\n- Store 5: 961 units\n- Store 6: 787 units\n- Store 7: 401 units\n- Store 8: 789 units\n\nSupply Capacity of Each Distribution Center (in units):\n- Center 1: 1365 units\n- Center 2: 1929 units\n- Center 3: 1502 units\n- Center 4: 1201 units\n- Center 5: 1529 units\n- Center 6: 961 units\n\nThe optimal solution must navigate the intricate balance between opening costs, transportation costs, supply capacities, and the demands of the retail stores. \n\nQuestion:\nWhat is the optimal total cost for GlobalSupply Inc. to open distribution centers and transport goods to meet all the retail stores' demands, considering the cost of opening the centers and the transportation costs of supplying the stores?\n", "en_answer": "585022.0"}
```

### Type of Correction: Correcting Answer
We invited several Operations Research experts to carefully review the remaining benchmark questions one by one. During this process, we identified some discrepancies in the provided answers. After verification, we have updated them with the correct values.

------

#### Traveling Salesman Problem (TSP)

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
More details are as follows:
| Index | Original Answer | Correct Answer |
|-------|-----------------|----------------|
| 59    | 213.0           | 245.0          |
| 60    | 60.0            | 121.0          |
| 61    | 142.0           | 165.0          |
| 62    | 50.0            | 127.0          |
| 64    | 194.0           | 213.0          |
| 65    | 206.0           | 232.0          |
| 66    | 228.0           | 233.0          |
| 68    | 158.0           | 175.0          |
| 69    | 162.0           | -9999          |
| 70    | 138.0           | 153.0          |
| 71    | 138.0           | 160.0          |
| 72    | 140.0           | 159.0          |
| 74    | 240.0           | 261.0          |
| 75    | 206.0           | 211.0          |
| 76    | 165.0           | 237.0          |
| 77    | 248.0           | 260.0          |
| 78    | 242.0           | 269.0          |
| 79    | 241.0           | 251.0          |
| 80    | 140.0           | 182.0          |
| 81    | 199.0           | 212.0          |
| 82    | 138.0           | 145.0          |
| 84    | 210.0           | 216.0          |
| 85    | 252.0           | 284.0          |
| 87    | 204.0           | 229.0          |
| 88    | 134.0           | 175.0          |
| 89    | 275.0           | 299.0          |
| 90    | 124.0           | 154.0          |
| 91    | 148.0           | 249.0          |
| 92    | 192.0           | 198.0          |
| 93    | 160.0           | 176.0          |
| 94    | 242.0           | 270.0          |
| 95    | 146.0           | 165.0          |
| 96    | 162.0           | 191.0          |
| 97    | 56.0            | 136.0          |
| 98    | 150.0           | 169.0          |



#### Other Questions with Incorrect Answers

------

For the other questions, we have updated them with the correct values.

The details are as follows:

| Index | Original Answer  | Correct Answer |
| ----- | ---------------- | -------------- |
| 177   | 242000           | 21600          |
| 183   | 2370             | 2400           |
| 185   | 8                | 10             |
| 186   | 9                | 11             |
| 196   | 4.596            | 66.366         |
| 198   | 0.6              | 1.0            |
| 199   | 5                | 7              |
| 200   | 103960           | 106380         |

### Conclusion and further work
We conducted a preliminary review and correction of the NL4OPT, IndustryOR, MAMO-ComplexLP, and MAMO-EasyLP datasets, detailing the process for the MAMO datasets. 
We encourage other researchers to use these revised versions for their future work and consider cite our paper. Furthermore, we plan to conduct a more comprehensive review and add additional instances to the IndustryOR dataset.

For any questions or issues regarding the datasets, please raise an issue on our GitHub repository or contact one of the authors via emails:
   * Yitian Chen, chenyitian@shanshu.ai
   * Minglong Cao, mlcao25@m.fudan.edu.cn
   * Siyu Shao, siyu_shao@connect.hku.hk
