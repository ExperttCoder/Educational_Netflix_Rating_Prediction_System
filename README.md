# Netflix Rating Prediction System

This project implements a movie rating prediction system using various machine learning algorithms, specifically designed to work with Netflix-style rating data. The system includes implementations of Expectation Maximization (EM) and K-means clustering algorithms for collaborative filtering.

## Project Structure

```
netflix/
├── algorithms/
│   ├── em.py
│   ├── naive_em.py
│   └── kmeans.py
├── data/
│   ├── netflix_complete.txt
│   ├── netflix_incomplete.txt
│   ├── test_complete.txt
│   ├── test_incomplete.txt
│   ├── test_solutions.txt
│   └── toy_data.txt
├── utils/
│   └── common.py
└── main.py
```

## Features

- Implementation of EM (Expectation Maximization) algorithm for collaborative filtering
- Implementation of K-means clustering for user grouping
- Naive EM implementation for comparison
- Common utility functions for data processing
- Test suite for algorithm validation
- Support for both complete and incomplete dataset processing

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Pandas

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/netflix-rating-prediction.git
cd netflix-rating-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To run the main prediction system:
```bash
python main.py
```

For testing:
```bash
python test.py
```

## Data Format

The system works with two types of data files:
- Complete data: Contains all ratings
- Incomplete data: Contains partial ratings for prediction

Data format example:
```
user_id,movie_id,rating
1,1,5
1,2,3
2,1,4
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on collaborative filtering techniques
- Inspired by the Netflix Prize competition 