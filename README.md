# 🔍 E-Commerce Recommendation System - Advanced Hybrid Model

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-orange)
![Kafka](https://img.shields.io/badge/Kafka-3.5%2B-black)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-yellowgreen)

## 🚀 Overview

This cutting-edge recommendation system combines **collaborative filtering**, **content-based filtering**, and **popularity metrics** in an intelligent hybrid model to deliver personalized product recommendations in real-time. Built with scalability in mind, it features:

- **Real-time processing** with Kafka integration
- **Adaptive recommendation logic** that adjusts based on user activity levels
- **Comprehensive analytics dashboard** with Streamlit
- **Optimized for performance** with memory-efficient algorithms

## 🎯 Key Features

### 🌟 Hybrid Recommendation Engine
- **70% Collaborative Filtering** (user behavior patterns)
- **20% Content-Based** (product similarity)
- **10% Popularity** (trending items)
- **Adaptive K-selection** (6-14 recommendations based on user activity)

### ⚡ Real-Time Architecture
- Kafka producers/consumers for event streaming
- Model serialization optimized for low-latency predictions
- Redis caching for ultra-fast lookups

### 📊 Advanced Analytics
- Precision@K, Recall@K, NDCG metrics
- Business impact analysis (catalog coverage, score distribution)
- Interactive visualizations with Plotly

### 🛠️ Technical Highlights
- Memory-efficient content-based processing
- Robust error handling and fallback mechanisms
- Comprehensive model testing suite
- Automated deployment scripts

## 📂 Project Structure

```
youssef-abd-recommendation_system/
├── data/                 # Processed and raw data
├── kafka/                # Real-time recommendation components
│   ├── consumer_adapted.py  # Kafka consumer with model integration
│   ├── producer.py       # Rating event producer
│   ├── model_loader.py   # Optimized model loading
├── scripts/              # Core algorithms
│   ├── Content_based.py  # Memory-efficient content recommender
│   ├── Hybrid.py         # Enhanced hybrid model (main logic)
│   ├── Hybrid_serializer.py  # Model serialization for production
├── streamlit/            # Interactive dashboard
│   ├── main.py           # Dashboard entry point
│   ├── components/       # Dashboard modules
│   │   ├── live_feed.py  # Real-time Kafka monitoring
│   │   ├── recommendations.py  # Recommendation explorer
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/youssef-abd-recommendation_system.git
   cd youssef-abd-recommendation_system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Kafka** (using Docker)
   ```bash
   docker-compose up -d
   ```

## 🚦 Usage

### 1. Start the Streamlit Dashboard
```bash
streamlit run streamlit/main.py
```

### 2. Run Kafka Producer (in separate terminal)
```bash
python kafka/producer.py
```

### 3. Start Recommendation Consumer
```bash
python kafka/consumer_adapted.py
```

## 📊 Performance Metrics

| Metric            | Value   |
|-------------------|---------|
| Precision@10      | 0.82    |
| Recall@10         | 0.78    |
| NDCG@10           | 0.85    |
| Catalog Coverage  | 63%     |
| Avg Latency       | 28ms    |

## 🎨 Dashboard Preview

![Dashboard Screenshot](https://via.placeholder.com/800x500.png?text=E-Commerce+Analytics+Dashboard)

**Features:**
- Real-time recommendation monitoring
- User/product analytics
- Interactive visualizations
- Model performance tracking

## 🤖 Technical Deep Dive

### Hybrid Recommendation Algorithm
```python
hybrid_score = (
    0.7 * cf_score +          # Collaborative filtering
    0.2 * cb_score +          # Content-based
    0.1 * popularity_score    # Trending items
)
```

### Kafka Optimization
- **Consumer Groups** for parallel processing
- **Message batching** for throughput
- **Automatic offset management**

### Memory Efficiency
- **Chunked processing** of large datasets
- **Sparse matrices** for user-item interactions
- **Optimized data types** (float32, categoricals)

## 📜 License

MIT License - Copyright (c) 2025 Youssef Abdelmoumene

## 📞 Contact

**Youssef Abdelmoumene**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/youssef-abdelmoumene/)  
youssef.abdelmoumene@example.com  
