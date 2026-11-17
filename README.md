# IA26_P01_E01 — Local Food Delivery Routing Agent

> Project 01 · Artificial Intelligence (AI) 2026-27
> Degree on Computer Systems Engineering · IPCA (Escola Superior de Tecnologia)

**Repository:** `IA26_P01_E01` (public GitHub repository)
**Repository Link:** <a href="https://github.com/OierZu/IA26_P01_E01.git">GitHub project</a>
**Status:** all three algorithm blocks implemented and executed (informed search, local search, CSP)

---

## 1. Team

| Name | Student number | GitHub user |
|------|----------------|-------------|
| Oier Zuñiga | 51240 | OierZu |
| Matei Panturu | 51235 | iv-panzer |
| Aime-Bruce Ishimwe | 51224 | bruceaime |

Lecturer: Joaquim Silva (added as collaborator of the repository).

---

## 2. Project overview

The goal is to **develop an intelligent agent**, documenting the design and implementation steps. The agent solves a routing problem for a local food distribution platform, using the AI techniques covered in the course: informed search, local search and constraint satisfaction.

### Theme: Local food

**Motivation**
- Demand for local food is growing in the EU, but small producers face high logistics costs and price volatility.
- Short supply chains reduce transport emissions and food waste compared to long-distance distribution.
- Fruit and vegetable prices have risen sharply in recent years, squeezing both producers and consumers.
- A platform that connects producers, consumers and logistics can make local distribution efficient and viable.
- **The key challenge is routing:** selecting efficient pickup-and-delivery routes under time-window constraints.

### Business model

A digital platform connecting three types of actors:

| Actor | Role |
|-------|------|
| **Producers** | Small local farms offering fruits, vegetables and other products |
| **Customers** | Place orders and choose a delivery location |
| **Collection centers (warehouses)** | Aggregate products from producers before distribution; operate a fleet of vehicles |

The platform's **routing agent** selects the best pickup-and-delivery routes by:
- Considering pickup locations (producers), drop-off locations (chosen by each customer) and the distances/times between all locations.
- Respecting the time windows in which products are available at the producer and must be delivered to the customer.

---

## 3. Algorithms applied

| Technique | Application in this project |
|-----------|-----------------------------|
| **Informed search** (Greedy best-first, A\*) | Find efficient pickup-and-delivery routes using a distance/time heuristic. Compared with 2 heuristics. |
| **Local search** (Hill climbing) | Improve an initial routing solution (the Greedy one) by iteratively exploring neighboring route configurations. |
| **Constraint Satisfaction Problems (CSP)** | Model time-window constraints (availability at producers, delivery deadlines at customers) and vehicle capacity as constraints to satisfy. |

---

## 4. Dataset

A **synthetic dataset**, generated as a set of CSV files (no database engine required). It covers the **Braga district** and **Póvoa de Varzim**.

It contains everything the routing agent needs: producers, collection centers (warehouses), vehicles, customers and delivery locations, customer orders with pickup and delivery time windows, and a distance-and-time table between every pair of routing stops.

Several datasets with the same structure are provided (via Moodle): `tiny`, `small`, and larger ones.

### 4.1 Entities and relationships

| Entity | Description |
|--------|-------------|
| **Locality** | Shared geocoded place (parish, municipality, district, coordinates) used by producers, centers and customers |
| **Producer** | Small local producer, located at a Locality, offering one or more Products |
| **Product** | Item offered by a Producer |
| **CollectionCenter** | Warehouse where goods are aggregated; home base for the Vehicles |
| **Vehicle** | Belongs to a CollectionCenter; has a capacity |
| **Customer** | Places CustomerOrders and chooses a DeliveryLocation for each one |
| **DeliveryLocation** | Delivery point chosen by the customer, located at a Locality |
| **CustomerOrder / OrderItem** | One order per customer; each item has an availability window at the producer and a delivery window at the delivery location |
| **Locations** | Derived routing layer built from the coordinates of producers, centers and delivery locations |
| **DistanceMatrix** | Derived routing layer with distance and travel time between every ordered pair of stops |

### 4.2 Orders and time windows

- Each customer order is linked to exactly one delivery location, chosen by the customer.
- Each order line (OrderItem) carries two time windows:
  - `availability_start` / `availability_end`: when the product can be collected at the producer.
  - `delivery_start` / `delivery_end`: when it must arrive at the delivery location.
- These windows are exactly the **constraints the routing agent must satisfy**: pickups cannot happen before a product is available, and deliveries cannot happen after the delivery window closes.

### 4.3 Distance and travel-time matrix

`distance_matrix` gives distance and travel time between every ordered pair of routing stops (producers, collection centers, delivery locations):

- `distance_km`: great-circle distance scaled by a route factor (default **1.3**) to approximate real road distance.
- `duration_min`: distance converted to time using an average road speed (default **45 km/h**).

It is a synthetic approximation, but deterministic and symmetric. It is enough to exercise informed search, local search and CSP time-window constraints without needing a live maps API.

---

## 5. Repository structure

```
IA26_P01_G##/
├── data_tiny/                 # CSV dataset (tiny)
├── data_small/                # CSV dataset (small)
├── data_*/                    # Other datasets
├── src/
│   ├── model.py               # Entities, cost function, route feasibility
│   ├── informed.py            # Greedy, A*, heuristics
│   ├── local_search.py        # Hill climbing
│   └── csp.py                 # Backtracking, forward checking, AC-3
├── notebooks/
│   ├── 01_Project01_G##.ipynb # Main project notebook
│   ├── 02_eda_tiny.ipynb      # Exploratory data analysis
│   └── 03_*.ipynb             # Other notebooks (informed, local search, CSP)
└── README.md
```

---

## 6. Notebook structure

The main notebook, `01_Project01_G##.ipynb`, is organized with each step in a distinct block, text blocks documenting reasoning, decisions, results and analysis, and brief comments in the code.

### Preliminary delivery

| Section | Content | Deliverable |
|---------|---------|-------------|
| **Introduction** | Context and purpose of the project; identification of the team | Main notebook |
| **Exploratory Data Analysis (EDA)** | Load CSVs and explore the data; validate integrity and time windows; measure weights vs. capacities; map stops | `02_eda_tiny.ipynb` |
| **Modeling** | Define `Problem`, `Request`, `Vehicle` and `State` classes; load instances from CSVs; implement the cost function and route feasibility checker | `model.py` |

### Final delivery

| Section | Content | Deliverable |
|---------|---------|-------------|
| **Informed search** | Greedy and A\* with 2 heuristics; routes for the 2 tiny vehicles (count expanded nodes) | `informed.py` notebook |
| **Local search** | Hill-Climbing on top of the Greedy solution; cost vs. iteration curves | `local_search.ipynb` |
| **CSP** | Variables, domains and constraints definition; heuristics applied to implement the soft constraints; feasible assignment and number of backtracks | `csp.ipynb` |
| **Results analysis** | Best solutions found; critical analysis; execution times; comparison between algorithms; future improvements for the agent | Main notebook |
| **Conclusion** | Outcomes, development process and tools used | Main notebook |

---

## 7. Progress

- [x] Public GitHub repository created and collaborators added
- [x] Dataset loaded and explored (EDA) — `notebooks/02_eda_tiny.ipynb`
- [x] Data integrity and time-window validation
- [x] Domain model (`Problem`, `Request`, `Vehicle`, `State`) — `src/model.py`
- [x] Cost function and route feasibility checker (tested with hand-made wrong routes) — `notebooks/03_modeling.ipynb`
- [x] Informed search (Greedy, A\*, heuristics h1 and h2) — `src/informed.py`, `notebooks/04_informed.ipynb`
- [x] Local search (Hill-Climbing from the Greedy solution) — `src/local_search.py`, `notebooks/05_local_search.ipynb`
- [x] CSP (backtracking, MRV, degree, LCV, forward checking, AC-3) — `src/csp.py`, `notebooks/06_csp.ipynb`
- [x] Main notebook with results analysis and conclusion — `notebooks/01_Project01_E01.ipynb`

## 8. How to run

```
pip install pandas numpy matplotlib jupyter
cd notebooks
jupyter notebook 01_Project01_E01.ipynb      # Run All (about 40 seconds)
```

The code uses only `pandas`, `numpy`, `matplotlib`, `heapq`, `time` and `itertools`.

## 9. Results (executed, tiny / small, total km)

| Method | tiny | small |
|--------|------|-------|
| Greedy h1 | 724.28 | 1090.63 |
| A\* (h1 and h2) | 724.28 | 1031.75 |
| Hill-Climbing from Greedy h1 | 580.90 | 785.50 |
| CSP + A\* legs | 662.47 | 867.50 |

All solutions pass the independent feasibility checker. Details, expanded nodes, backtracks and times are in the main notebook.

## 10. Implementation decisions to confirm with the lecturer

The brief leaves several points open. Section 3.1 of the main notebook lists what we assumed (waiting allowed, service per stop, return to the center, one visit per producer and day, legs deliver-stored / collect / deliver-new per vehicle and day, refrigeration as a soft preference only).
