# Educational Netflix Rating Prediction System

A sophisticated machine learning project implementing collaborative filtering algorithms to predict user ratings for movies, inspired by the Netflix Prize competition. This educational project demonstrates the implementation of various machine learning algorithms for recommendation systems.

## 🎯 Overview

This system implements multiple approaches to solve the movie rating prediction problem:
- Expectation Maximization (EM) Algorithm
- K-means Clustering
- Naive EM Implementation
- Collaborative Filtering Techniques

## 🏗️ Project Structure

```
netflix/
├── algorithms/          # Core ML algorithm implementations
│   ├── em.py           # Expectation Maximization algorithm
│   ├── naive_em.py     # Naive implementation of EM
│   └── kmeans.py       # K-means clustering implementation
├── data/               # Dataset files
│   ├── netflix_complete.txt
│   ├── netflix_incomplete.txt
│   ├── test_complete.txt
│   ├── test_incomplete.txt
│   ├── test_solutions.txt
│   └── toy_data.txt
├── utils/              # Utility functions
│   └── common.py       # Common helper functions
└── main.py            # Main application entry point
```

## ✨ Features

- **Advanced ML Algorithms**: 
  - Expectation Maximization (EM) for collaborative filtering
  - K-means clustering for user grouping
  - Naive EM implementation for educational comparison
  
- **Comprehensive Testing**:
  - Validation suite for algorithm accuracy
  - Performance benchmarking tools
  - Test datasets included

- **Data Processing**:
  - Support for complete and incomplete datasets
  - Efficient data handling mechanisms
  - Flexible data format support

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ExperttCoder/Educational_Netflix_Rating_Prediction_System.git
cd Educational_Netflix_Rating_Prediction_System
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Usage

1. Run the main prediction system:
```bash
python main.py
```

2. For testing the implementation:
```bash
python test.py
```

## 📊 Data Format

The system works with two types of data files:

1. **Complete Data Format**:
```
user_id,movie_id,rating
1,1,5
1,2,3
2,1,4
```

2. **Incomplete Data Format** (for prediction):
```
user_id,movie_id,rating
1,1,?
1,2,3
2,1,?
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on collaborative filtering techniques
- Inspired by the Netflix Prize competition
- Educational implementation for learning purposes

## 📚 References

- Netflix Prize Competition
- Collaborative Filtering Research Papers
- Machine Learning Algorithms Documentation

## 🔗 Links

- [Project Repository](https://github.com/ExperttCoder/Educational_Netflix_Rating_Prediction_System)
- [Issue Tracker](https://github.com/ExperttCoder/Educational_Netflix_Rating_Prediction_System/issues)
- [MIT License](LICENSE) 