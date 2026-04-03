# c5_Q-state.csv Description

## Overview
The `c5_Q-state.csv` dataset is a quantum logic representation of quantum values, 1-39, presenting across 5 Quantum states (QS) and multiple events.

## Dataset Structure

### Format
- **File Type**: CSV (Comma Separated Values)
- **Total Events**: Approximately 11,756 entries
- **Encoding**: ASCII/UTF-8 compatible
- **Memory footprint**: ~4.0 MB

### Columns
1. **date** (Column 1): Unique identifier for each event in sequential numbers (e.g., 11448, 11449, etc.)
2. **Quantum States (QS_x)** (Columns 2-6): Five integer values representing the quantum states (QS_1 through QS_5) for each event, with no duplicate quantum values 1-39 in a single event. The QS values are in ascending order.
3. **Header Row**
date,QS_1,QS_2,QS_3,QS_4,QS_5

## Data Characteristics

## Cylindrical Adjacency
The dataset exhibits cylindrical adjacency, meaning Q-values, 1-39, positions wrap around (position 39 is adjacent to position 1), creating a cylindrical representation of the quantum value space.

