# Customer Segmentation Using Machine Learning

A machine learning project that groups customers based on their annual income and spending score using K-Means clustering. The trained model is deployed as a REST API using FastAPI.

## Project Overview

Customer segmentation helps businesses understand different customer groups and develop better marketing strategies. This project uses unsupervised machine learning to identify customer segments from the Mall Customers dataset.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- K-Means Clustering
- FastAPI
- Uvicorn
- Joblib
- Render (deployment)

## Dataset

The project uses the Mall Customers dataset, which contains customer demographic and spending information.

Features used for clustering:
- Annual Income (k$)
- Spending Score (1–100)

## Machine Learning Workflow

1. Load and explore the dataset.
2. Perform exploratory data analysis and visualize customer patterns.
3. Scale the selected features using StandardScaler.
4. Apply K-Means clustering to group customers.
5. Evaluate clustering results using the silhouette score and visualizations.
6. Save the trained scaler and K-Means model.
7. Load the saved models in a FastAPI application.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Check whether the API is running |
| GET | `/health` | Health check, if enabled in the application |
| POST | `/predict` | Predict a customer's cluster |

## Try the Live API

- **API URL:** YOUR-RENDER-URL
- **Interactive documentation:** YOUR-RENDER-URL/docs

In the interactive documentation, expand `POST /predict`, click **Try it out**, and submit:

```json
{
  "annual_income": 60,
  "spending_score": 75
}
```

The API returns the input values and the predicted cluster number.

## Run Locally

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Start the API:

```bash
uvicorn main:app --reload
```

5. Open `http://127.0.0.1:8000/docs` in your browser.

Ensure `scaler.pkl` and `kmeans_model.pkl` are present in the project directory before starting the API.

## Project Structure

```text
customer-segmentation-api/
├── main.py
├── customer_segmentation.ipynb
├── scaler.pkl
├── kmeans_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## Future Improvements

- Build an interactive customer segmentation dashboard.
- Improve cluster interpretation and visualization.
- Add a user-friendly interface for customer predictions.

## Author

Neeraj Parimi
