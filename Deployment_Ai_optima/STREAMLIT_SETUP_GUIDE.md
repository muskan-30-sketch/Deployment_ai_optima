# Streamlit Deployment Guide
## Dynamic Pricing System - ML-Based Price Optimization

---

## 📋 TABLE OF CONTENTS

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [File Structure](#file-structure)
4. [Running the Application](#running-the-application)
5. [Features Overview](#features-overview)
6. [Deployment Options](#deployment-options)
7. [Troubleshooting](#troubleshooting)

---

## 📦 PREREQUISITES

- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for version control)
- ~500MB disk space for dependencies

### System Requirements
- CPU: Dual-core or higher (recommended)
- RAM: 4GB minimum (8GB recommended)
- Internet: Required for initial setup

---

## 🚀 INSTALLATION

### Step 1: Create Project Directory

```bash
# Create a new directory for your project
mkdir dynamic-pricing-system
cd dynamic-pricing-system
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Check if Streamlit is installed
streamlit --version

# Check if models can be imported
python -c "import xgboost as xgb; import lightgbm as lgb; print('✓ All libraries installed')"
```

---

## 📁 FILE STRUCTURE

Your project directory should look like this:

```
dynamic-pricing-system/
├── streamlit_app.py                    # Main Streamlit application
├── requirements.txt                    # Python dependencies
├── clean_dataset_numeric.csv           # Training dataset
├── xgboost_revenue_model.pkl          # Trained XGBoost model
├── lightgbm_revenue_model.pkl         # Trained LightGBM model
├── feature_columns.pkl                # Feature column names
├── milestone5_results_summary.json    # Model metrics
└── README.md                          # This file

# Optional: Sample data for batch processing
├── sample_batch.csv
└── sample_predictions.csv
```

### File Descriptions

| File | Purpose | Created By |
|------|---------|-----------|
| streamlit_app.py | Main web application | You (this guide) |
| requirements.txt | Dependencies list | You (this guide) |
| clean_dataset_numeric.csv | Training/validation data | Milestone 4 |
| xgboost_revenue_model.pkl | XGBoost model | Milestone 5 notebook |
| lightgbm_revenue_model.pkl | LightGBM model | Milestone 5 notebook |
| feature_columns.pkl | Feature names | Milestone 5 notebook |

---

## ▶️ RUNNING THE APPLICATION

### Basic Usage

```bash
# Navigate to project directory
cd dynamic-pricing-system

# Activate virtual environment (if created)
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Run Streamlit app
streamlit run streamlit_app.py
```

### Expected Output

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### Access the Application

Open your web browser and go to:
- **Local**: http://localhost:8501
- **Network**: http://YOUR_IP:8501

---

## 🎯 FEATURES OVERVIEW

### 📊 Dashboard
- **Purpose**: View overall model performance
- **Features**:
  - Real-time model metrics (R², MAE, RMSE)
  - XGBoost vs LightGBM comparison
  - Actual vs Predicted plots
  - Error distribution analysis
  - Residuals visualization

### 🔮 Price Prediction
- **Purpose**: Predict revenue for individual products
- **Features**:
  - Input product details
  - Get predictions from both models
  - View ensemble predictions
  - See revenue breakdown
  - Compare actual vs predicted

### ⚡ Price Optimization
- **Purpose**: Find optimal prices to maximize revenue
- **Features**:
  - Automatic price optimization
  - Customizable discount ranges
  - Price multiplier settings
  - Revenue lift calculation
  - Download optimization results
  - Before/after comparison

### 📈 Performance Analysis
- **Purpose**: Deep dive into model metrics
- **Features**:
  - Group analysis (by month, quarter, product, etc.)
  - Metric selection (R², MAE, RMSE, MAPE)
  - Performance trends
  - Visualization by categories
  - Export analysis results

### 💼 Batch Processing
- **Purpose**: Process multiple products at once
- **Features**:
  - Upload CSV files
  - Process all products simultaneously
  - Generate batch predictions
  - Error analysis
  - Download batch results
  - Visualization of results

### ⚙️ Settings & Configuration
- **Purpose**: Configure system parameters
- **Features**:
  - System status monitoring
  - Model information
  - Feature documentation
  - Advanced configuration
  - Performance benchmarks
  - User guide

---

## 🌐 DEPLOYMENT OPTIONS

### Option 1: Local Deployment (Development)

```bash
streamlit run streamlit_app.py
```

**Best For**: Testing, development, small teams
**Pros**: Quick setup, no infrastructure needed
**Cons**: Limited to local network, not production-grade

### Option 2: Streamlit Cloud (Recommended for Beginners)

1. Push code to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git push origin main
```

2. Go to https://share.streamlit.io
3. Sign in with GitHub
4. Click "New app"
5. Select your repository and branch
6. Streamlit deploys automatically

**Best For**: Public demos, prototypes, learning
**Pros**: Free, easy setup, automatic updates
**Cons**: Public app, limited resources

### Option 3: Heroku Deployment

Create `Procfile`:
```
web: streamlit run streamlit_app.py --logger.level=error
```

Create `.streamlit/config.toml`:
```toml
[server]
port = $PORT
enableCORS = false
```

Deploy:
```bash
heroku login
heroku create your-app-name
git push heroku main
```

**Best For**: Production apps, custom domain
**Pros**: Reliable, scalable, custom domain support
**Cons**: Paid, requires setup

### Option 4: AWS EC2 Deployment

1. Launch EC2 instance (Ubuntu 20.04)
2. SSH into instance
3. Install Python and dependencies:
```bash
sudo apt-get update
sudo apt-get install python3-pip
pip3 install -r requirements.txt
```

4. Run Streamlit:
```bash
streamlit run streamlit_app.py --server.port 8501
```

**Best For**: Enterprise, high traffic
**Pros**: Full control, scalable, secure
**Cons**: Complex setup, requires AWS knowledge

### Option 5: Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]
```

Build and run:
```bash
docker build -t pricing-system .
docker run -p 8501:8501 pricing-system
```

**Best For**: Containerized deployments, consistent environments
**Pros**: Consistent across systems, easy scaling
**Cons**: Requires Docker knowledge

---

## 🔧 CONFIGURATION

### Streamlit Configuration File

Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#f0f2f6"
secondaryBackgroundColor = "#e8eaf6"
textColor = "#31333F"

[server]
port = 8501
enableCORS = false
enableXsrfProtection = true
headless = true

[logger]
level = "info"

[client]
showErrorDetails = true
```

### Custom Styling

Modify colors in `streamlit_app.py`:
```python
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        ...
    }
    </style>
    """, unsafe_allow_html=True)
```

---

## 🐛 TROUBLESHOOTING

### Problem: "ModuleNotFoundError: No module named 'streamlit'"

**Solution**:
```bash
pip install streamlit
# Or reinstall all dependencies
pip install -r requirements.txt
```

### Problem: "FileNotFoundError: xgboost_revenue_model.pkl"

**Solution**:
1. Make sure the model files are in the same directory as `streamlit_app.py`
2. Check file names match exactly:
   - `xgboost_revenue_model.pkl`
   - `lightgbm_revenue_model.pkl`
   - `feature_columns.pkl`

### Problem: "Port 8501 already in use"

**Solution**:
```bash
# Use a different port
streamlit run streamlit_app.py --server.port 8502

# Or kill the process using port 8501
# Windows: netstat -ano | findstr :8501
# Linux/Mac: lsof -i :8501
```

### Problem: "ImportError: No module named 'plotly'"

**Solution**:
```bash
pip install plotly
```

### Problem: CSV file not loading in batch processing

**Solution**:
1. Check CSV format:
   - Must have columns: UnitPrice, Discount, Quantity, OrderDate, TotalAmount
   - No special characters in headers
   - Proper date format (YYYY-MM-DD)

2. Test the CSV:
```python
import pandas as pd
df = pd.read_csv('your_file.csv')
print(df.head())
print(df.info())
```

### Problem: Slow performance or timeouts

**Solution**:
1. Reduce sample size in optimization
2. Cache data more aggressively:
```python
@st.cache_data(ttl=3600)
def load_data(csv_path):
    ...
```

3. Use smaller subsets for analysis

---

## 📊 EXAMPLE WORKFLOWS

### Workflow 1: Single Product Price Prediction

1. Navigate to "Price Prediction" page
2. Enter product details:
   - Unit Price: $100
   - Discount: 5%
   - Quantity: 5
   - Select date/time
3. Click "Predict Revenue"
4. View predictions from both models
5. Compare with ensemble prediction

### Workflow 2: Optimize 50 Products

1. Go to "Price Optimization" page
2. Adjust settings:
   - Price multiplier: 0.8 - 1.3
   - Discount range: 0% - 30%
   - Sample size: 50
3. Click "Run Optimization"
4. Review results and revenue lift
5. Download CSV for implementation

### Workflow 3: Analyze Monthly Performance

1. Navigate to "Performance Analysis"
2. Select:
   - Metric: R² Score
   - Group by: Month
3. Run analysis
4. View performance by month
5. Identify seasonal patterns
6. Download insights

### Workflow 4: Process Batch of 1000 Orders

1. Go to "Batch Processing"
2. Upload CSV with 1000 products
3. Click "Process Batch"
4. Wait for completion
5. Review summary metrics
6. Download predictions and errors
7. Analyze results

---

## 📈 MONITORING & MAINTENANCE

### Daily Tasks
- Check dashboard for anomalies
- Verify prediction accuracy
- Monitor system performance

### Weekly Tasks
- Review batch processing results
- Analyze performance trends
- Update feature data

### Monthly Tasks
- Retrain models with new data
- Review and update pricing strategies
- Generate performance reports

### Quarterly Tasks
- Full model evaluation
- Feature importance analysis
- Strategy optimization

---

## 🔒 SECURITY CONSIDERATIONS

### When Deploying to Production

1. **Authentication**:
   - Use Streamlit authentication
   - Implement user access controls
   - Protect sensitive data

2. **Data Protection**:
   - Use HTTPS only
   - Encrypt sensitive data
   - Implement access logging

3. **API Security**:
   - Rate limiting
   - Input validation
   - Error handling

4. **Monitoring**:
   - System logging
   - Performance monitoring
   - Alert systems

---

## 📞 SUPPORT & RESOURCES

### Documentation
- [Streamlit Docs](https://docs.streamlit.io)
- [Plotly Docs](https://plotly.com/python/)
- [XGBoost Docs](https://xgboost.readthedocs.io/)
- [LightGBM Docs](https://lightgbm.readthedocs.io/)

### Community
- [Streamlit Forum](https://discuss.streamlit.io)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/streamlit)
- [GitHub Discussions](https://github.com/streamlit/streamlit/discussions)

### Getting Help
1. Check the Troubleshooting section
2. Review logs: `streamlit logs`
3. Check model file integrity
4. Verify data format
5. Test with sample data

---

## 🎓 NEXT STEPS

1. **Test Locally**: Run the app and test all features
2. **Verify Models**: Ensure predictions are reasonable
3. **Add Custom Data**: Test with your actual data
4. **Deploy**: Choose deployment option and deploy
5. **Monitor**: Track performance and collect feedback
6. **Iterate**: Update models and features based on results

---

## 📝 CHANGELOG

### Version 1.0 (Initial Release)
- ✅ Dashboard with model comparison
- ✅ Single product price prediction
- ✅ Batch price optimization
- ✅ Performance analysis
- ✅ Batch processing
- ✅ Settings and configuration
- ✅ Model management

### Planned Features
- [ ] API endpoints
- [ ] User authentication
- [ ] Data persistence (database)
- [ ] Real-time monitoring
- [ ] Advanced analytics
- [ ] Model retraining interface
- [ ] A/B testing framework

---

## 📄 LICENSE

This project is part of AI: PriceOptima - Milestone 5

---

**Created**: 2024
**Last Updated**: 2024
**Version**: 1.0

For questions or issues, please refer to the troubleshooting section or contact your project mentor.
