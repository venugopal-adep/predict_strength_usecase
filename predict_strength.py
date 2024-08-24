import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(page_title="Strength Prediction App", layout="wide")

# Custom CSS for UI enhancement
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f8ff;
    }
    .stButton>button {
        background-color: #4682b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    .stSelectbox {
        background-color: #e6f2ff;
    }
    .stTabs>div {
        background-color: #ffffff;
        border-radius: 5px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load and preprocess the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('Izod.csv')
    df.columns = ['Lattice', 'Layer_height', 'Strength']
    df['Strength'] = pd.to_numeric(df['Strength'], errors='coerce')
    return df

df = load_data()

# Title and description
st.title("🔬 3D Printing Strength Prediction")
st.write("""
    Welcome to the 3D Printing Strength Prediction App! This tool helps you understand and predict the strength of 3D printed objects based on their lattice structure and layer height.
    
    **How it works:**
    1. Explore the dataset to see how lattice structure and layer height affect strength.
    2. Visualize correlations between features.
    3. Choose a prediction algorithm to build a model.
    4. Evaluate the model's performance and make predictions.
    
    Let's dive in and unlock the secrets of stronger 3D prints! 🚀
""")

# Sidebar for selecting algorithm and user input
st.sidebar.header("Model Settings")
algorithm = st.sidebar.selectbox(
    "Select the Prediction Algorithm",
    ("Linear Regression", "Decision Tree", "Random Forest")
)

st.sidebar.header("Make a Prediction")
user_lattice = st.sidebar.slider("Lattice Structure", float(df['Lattice'].min()), float(df['Lattice'].max()), float(df['Lattice'].mean()))
user_layer_height = st.sidebar.slider("Layer Height", float(df['Layer_height'].min()), float(df['Layer_height'].max()), float(df['Layer_height'].mean()))

# Prepare data
X = df[['Lattice', 'Layer_height']]
y = df['Strength']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dataset", "🔗 Correlation", "🎯 Feature Importance", "🤖 Model", "📈 Metrics"])

with tab1:
    st.header("Dataset Overview")
    st.dataframe(df.style.highlight_max(axis=0))
    
    st.subheader("Data Distribution")
    fig = px.scatter_3d(df, x='Lattice', y='Layer_height', z='Strength', color='Strength',
                        title="3D Scatter Plot of Lattice, Layer Height, and Strength")
    st.plotly_chart(fig)

with tab2:
    st.header("Correlation Matrix")
    correlation_matrix = df.corr()
    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r")
    fig.update_layout(title="Feature Correlation Heatmap")
    st.plotly_chart(fig)

    st.write("""
    **Understanding Correlations:**
    - A value close to 1 indicates a strong positive correlation.
    - A value close to -1 indicates a strong negative correlation.
    - A value close to 0 indicates little to no correlation.
    
    For example, if Lattice and Strength have a correlation of 0.8, it means as the Lattice value increases, Strength tends to increase as well.
    """)

with tab3:
    st.header("Feature Importance")
    
    if algorithm in ["Decision Tree", "Random Forest"]:
        if algorithm == "Decision Tree":
            model = DecisionTreeRegressor()
        else:
            model = RandomForestRegressor()
        
        model.fit(X_train, y_train)
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
        fig = px.bar(feature_importance_df, x='Feature', y='Importance', title=f"Feature Importance for {algorithm}")
        st.plotly_chart(fig)
        
        st.write("""
        **Interpreting Feature Importance:**
        - The higher the bar, the more important the feature is for predicting strength.
        - For example, if Lattice has a higher importance, it means changes in the Lattice structure have a bigger impact on the final strength of the 3D printed object.
        """)
    else:
        st.write("Feature importance is not available for Linear Regression. Try selecting Decision Tree or Random Forest to see feature importance.")

with tab4:
    st.header("Model Visualization")
    
    if algorithm == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)

        intercept = model.intercept_
        coefficients = model.coef_
        st.subheader("Linear Regression Model Equation")
        st.write(f"Strength = {intercept:.2f} + ({coefficients[0]:.2f} * Lattice) + ({coefficients[1]:.2f} * Layer Height)")
        
        st.write("""
        **Interpreting the Equation:**
        - The intercept (%.2f) is the base strength when both Lattice and Layer Height are 0.
        - For every 1 unit increase in Lattice, the strength changes by %.2f units.
        - For every 1 unit increase in Layer Height, the strength changes by %.2f units.
        """ % (intercept, coefficients[0], coefficients[1]))
        
        # 3D plot for Linear Regression
        x_range = np.linspace(X['Lattice'].min(), X['Lattice'].max(), 50)
        y_range = np.linspace(X['Layer_height'].min(), X['Layer_height'].max(), 50)
        X_mesh, Y_mesh = np.meshgrid(x_range, y_range)
        Z = model.predict(np.c_[X_mesh.ravel(), Y_mesh.ravel()]).reshape(X_mesh.shape)

        fig = go.Figure(data=[
            go.Surface(x=x_range, y=y_range, z=Z),
            go.Scatter3d(x=X['Lattice'], y=X['Layer_height'], z=y, mode='markers', marker=dict(size=4))
        ])
        fig.update_layout(scene=dict(xaxis_title='Lattice', yaxis_title='Layer Height', zaxis_title='Strength'),
                          title="Linear Regression 3D Surface Plot")
        st.plotly_chart(fig)

    elif algorithm == "Decision Tree":
        model = DecisionTreeRegressor(max_depth=3)  # Limit depth for visualization
        model.fit(X_train, y_train)

        # Display decision tree plot
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(model, filled=True, feature_names=X.columns, rounded=True, ax=ax)
        st.pyplot(fig)
        
        st.write("""
        **Reading the Decision Tree:**
        - Each node shows a decision based on a feature and its value.
        - The leaf nodes (at the bottom) show the predicted strength.
        - Following a path from root to leaf shows how the model makes decisions.
        """)

    elif algorithm == "Random Forest":
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        
        st.write("""
        Random Forest combines multiple decision trees to make predictions. 
        While we can't visualize all trees, we can show feature importance and partial dependence plots.
        """)
        
        # Partial dependence plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(X['Lattice'].sort_values(), model.predict(pd.DataFrame({'Lattice': X['Lattice'].sort_values(), 'Layer_height': X['Layer_height'].mean()})))
        ax1.set_title('Partial Dependence on Lattice')
        ax1.set_xlabel('Lattice')
        ax1.set_ylabel('Predicted Strength')
        
        ax2.plot(X['Layer_height'].sort_values(), model.predict(pd.DataFrame({'Lattice': X['Lattice'].mean(), 'Layer_height': X['Layer_height'].sort_values()})))
        ax2.set_title('Partial Dependence on Layer Height')
        ax2.set_xlabel('Layer Height')
        ax2.set_ylabel('Predicted Strength')
        
        st.pyplot(fig)
        
        st.write("""
        **Interpreting Partial Dependence Plots:**
        - These plots show how changes in one feature affect the predicted strength, while keeping other features constant.
        - A steeper line indicates that the feature has a stronger effect on the prediction.
        """)

    # Make a prediction
    user_input = pd.DataFrame({'Lattice': [user_lattice], 'Layer_height': [user_layer_height]})
    prediction = model.predict(user_input)[0]
    
    st.subheader("Your Prediction")
    st.write(f"For a 3D print with Lattice {user_lattice:.2f} and Layer Height {user_layer_height:.2f}, the predicted Strength is: {prediction:.2f}")

with tab5:
    st.header("Model Evaluation Metrics")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    col2.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
    col3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
    col4.metric("R2 Score", f"{r2:.2f}")

    st.write("""
    **Understanding the Metrics:**
    - **MAE:** On average, our predictions are off by this amount.
    - **MSE:** Measures the average squared difference between predicted and actual values.
    - **RMSE:** Similar to MAE, but penalizes large errors more. It's in the same unit as the target variable.
    - **R2 Score:** Indicates how well the model explains the variability in the data. 1 is perfect, 0 means the model just predicts the mean.
    
    A lower MAE, MSE, and RMSE indicate better model performance. An R2 score closer to 1 is better.
    """)

    # Actual vs Predicted plot
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Strength', 'y': 'Predicted Strength'},
                     title=f"{algorithm}: Actual vs Predicted")
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                             mode='lines', name='Perfect Prediction'))
    st.plotly_chart(fig)

    st.write("""
    **Interpreting the Actual vs Predicted Plot:**
    - Points closer to the diagonal line indicate more accurate predictions.
    - Points above the line are overestimations, below are underestimations.
    - A good model will have points clustered tightly around the diagonal line.
    """)

st.sidebar.markdown("---")
