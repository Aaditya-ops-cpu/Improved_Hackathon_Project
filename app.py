import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import json
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="CredTech Credit Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .feature-importance {
        background-color: #fff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class CreditIntelligencePlatform:
    def __init__(self):
        self.companies = {
            'AAPL': 'Apple Inc.',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'AMZN': 'Amazon.com Inc.',
            'JPM': 'JPMorgan Chase & Co.',
            'JNJ': 'Johnson & Johnson',
            'XOM': 'Exxon Mobil Corporation',
            'WMT': 'Walmart Inc.'
        }
        self.feature_descriptions = {
            'debt_to_equity': 'Debt-to-Equity Ratio',
            'current_ratio': 'Current Ratio',
            'roa': 'Return on Assets',
            'profit_margin': 'Net Profit Margin',
            'revenue_growth': 'Revenue Growth (YoY)',
            'volatility': 'Stock Price Volatility',
            'sentiment_score': 'News Sentiment Score',
            'macroecon_index': 'Macroeconomic Index'
        }
        
    def generate_sample_data(self):
        """Generate synthetic financial data for demonstration"""
        np.random.seed(42)
        data = []
        
        for ticker, name in self.companies.items():
            for i in range(100):  
                date = datetime.now() - timedelta(days=100-i)
                
                base_debt_equity = np.random.uniform(0.5, 2.0)
                base_current_ratio = np.random.uniform(1.2, 3.0)
                base_roa = np.random.uniform(0.02, 0.15)
                base_profit_margin = np.random.uniform(0.05, 0.25)
                
                debt_to_equity = base_debt_equity + np.random.normal(0, 0.1)
                current_ratio = base_current_ratio + np.random.normal(0, 0.2)
                roa = max(0.01, base_roa + np.random.normal(0, 0.02))
                profit_margin = max(0.01, base_profit_margin + np.random.normal(0, 0.03))
                revenue_growth = np.random.normal(0.08, 0.05)
                volatility = np.random.uniform(0.1, 0.4)
                
                sentiment_score = np.random.normal(0.6, 0.3)
                sentiment_score = max(-1, min(1, sentiment_score))
                
                macroecon_index = np.random.normal(100, 10)
                
                features = [debt_to_equity, current_ratio, roa, profit_margin, 
                           revenue_growth, volatility, sentiment_score, macroecon_index]
                
                credit_score = self.calculate_credit_score(features)
                
                data.append({
                    'date': date,
                    'ticker': ticker,
                    'company_name': name,
                    'debt_to_equity': debt_to_equity,
                    'current_ratio': current_ratio,
                    'roa': roa,
                    'profit_margin': profit_margin,
                    'revenue_growth': revenue_growth,
                    'volatility': volatility,
                    'sentiment_score': sentiment_score,
                    'macroecon_index': macroecon_index,
                    'credit_score': credit_score
                })
        
        return pd.DataFrame(data)
    
    def calculate_credit_score(self, features):
        """Calculate credit score based on weighted features"""
        weights = [-0.15, 0.12, 0.18, 0.16, 0.14, -0.10, 0.08, 0.07]
        
        normalized_features = []
        for i, feature in enumerate(features):
            if i == 0:  
                normalized = max(0, 1 - feature / 3.0)
            elif i == 1:  
                normalized = min(1, max(0, (feature - 1) / 2))
            elif i == 5:  
                normalized = max(0, 1 - feature / 0.5)
            elif i == 6:  
                normalized = (feature + 1) / 2
            else: 
                normalized = min(1, max(0, feature))
            
            normalized_features.append(normalized)
        
        base_score = sum(w * f for w, f in zip(weights, normalized_features))
        credit_score = max(300, min(850, 300 + base_score * 550))
        
        return credit_score
    
    def train_credit_model(self, df):
        """Train an explainable credit model"""
        features = ['debt_to_equity', 'current_ratio', 'roa', 'profit_margin', 
                   'revenue_growth', 'volatility', 'sentiment_score', 'macroecon_index']
        
        def get_rating(score):
            if score >= 750:
                return 'Excellent'
            elif score >= 700:
                return 'Good'
            elif score >= 650:
                return 'Fair'
            else:
                return 'Poor'
        
        df['rating'] = df['credit_score'].apply(get_rating)
        
        X = df[features]
        y = df['rating']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt_model.fit(X_train, y_train)
        
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        dt_accuracy = accuracy_score(y_test, dt_model.predict(X_test))
        rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
        
        return dt_model, rf_model, dt_accuracy, rf_accuracy, features
    
    def get_feature_importance(self, model, features):
        """Get feature importance from trained model"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            importance = np.zeros(len(features))
            for i, feature in enumerate(features):
                importance[i] = np.random.uniform(0.05, 0.2)  # Placeholder
        
        return pd.DataFrame({
            'feature': features,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def generate_explanation(self, company_data, feature_importance):
        """Generate human-readable explanation for credit score"""
        latest = company_data.iloc[-1]
        
        explanations = []
        
        if latest['debt_to_equity'] > 1.5:
            explanations.append(f"High debt-to-equity ratio ({latest['debt_to_equity']:.2f}) indicates elevated financial risk")
        elif latest['debt_to_equity'] < 0.5:
            explanations.append(f"Conservative debt levels ({latest['debt_to_equity']:.2f}) support credit strength")
        
        if latest['roa'] > 0.1:
            explanations.append(f"Strong return on assets ({latest['roa']:.1%}) demonstrates efficient capital allocation")
        elif latest['roa'] < 0.03:
            explanations.append(f"Low return on assets ({latest['roa']:.1%}) may indicate operational challenges")
        
        if latest['sentiment_score'] > 0.7:
            explanations.append("Positive market sentiment supporting credit profile")
        elif latest['sentiment_score'] < 0.3:
            explanations.append("Negative market sentiment posing potential credit risks")
        
        if latest['revenue_growth'] > 0.1:
            explanations.append(f"Strong revenue growth ({latest['revenue_growth']:.1%}) enhances creditworthiness")
        elif latest['revenue_growth'] < 0:
            explanations.append(f"Revenue decline ({latest['revenue_growth']:.1%}) raises concerns about business momentum")
        
        top_feature = feature_importance.iloc[0]
        feature_name = self.feature_descriptions.get(top_feature['feature'], top_feature['feature'])
        explanations.append(f"{feature_name} is the most significant factor in the current assessment")
        
        return explanations
    
    def simulate_news_events(self, ticker):
        """Simulate recent news events for a company"""
        events = {
            'AAPL': [
                {"date": datetime.now() - timedelta(days=2), 
                 "event": "New product launch announced", "impact": "positive"},
                {"date": datetime.now() - timedelta(days=5), 
                 "event": "Supply chain issues reported", "impact": "negative"}
            ],
            'MSFT': [
                {"date": datetime.now() - timedelta(days=1), 
                 "event": "Cloud revenue exceeds expectations", "impact": "positive"},
                {"date": datetime.now() - timedelta(days=7), 
                 "event": "Regulatory scrutiny increasing", "impact": "negative"}
            ],
            'JPM': [
                {"date": datetime.now() - timedelta(days=3), 
                 "event": "Strong quarterly earnings reported", "impact": "positive"},
                {"date": datetime.now() - timedelta(days=10), 
                 "event": "Interest margin pressure", "impact": "negative"}
            ]
        }
        
        return events.get(ticker, [])

def main():
    platform = CreditIntelligencePlatform()
    
    st.markdown('<div class="main-header">🏦 CredTech Explainable Credit Intelligence Platform</div>', 
                unsafe_allow_html=True)
    
    with st.spinner('Loading financial data and training models...'):
        df = platform.generate_sample_data()
        dt_model, rf_model, dt_accuracy, rf_accuracy, features = platform.train_credit_model(df)
        feature_importance = platform.get_feature_importance(dt_model, features)
    
    st.sidebar.title("Navigation")
    
    selected_ticker = st.sidebar.selectbox(
        "Select Company",
        options=list(platform.companies.keys()),
        format_func=lambda x: f"{x} - {platform.companies[x]}"
    )
    
    date_range = st.sidebar.selectbox(
        "Time Period",
        ["Last 30 days", "Last 90 days", "Last 180 days", "All available"]
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    company_data = df[df['ticker'] == selected_ticker].copy()
    latest_data = company_data.iloc[-1]
    
    with col1:
        st.metric(
            label="Current Credit Score",
            value=f"{latest_data['credit_score']:.0f}",
            delta=f"{(latest_data['credit_score'] - company_data.iloc[-2]['credit_score']):.1f}"
        )
    
    with col2:
        rating = 'Excellent' if latest_data['credit_score'] >= 750 else \
                'Good' if latest_data['credit_score'] >= 700 else \
                'Fair' if latest_data['credit_score'] >= 650 else 'Poor'
        
        st.metric(
            label="Credit Rating",
            value=rating
        )
    
    with col3:
        st.metric(
            label="Model Accuracy",
            value=f"{(dt_accuracy * 100):.1f}%"
        )
    
    with col4:
        sentiment_trend = "Positive" if latest_data['sentiment_score'] > 0.6 else \
                         "Neutral" if latest_data['sentiment_score'] > 0.4 else "Negative"
        st.metric(
            label="Market Sentiment",
            value=sentiment_trend
        )
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Score Trends", "🔍 Feature Analysis", "📊 Model Insights", "🚨 Recent Events"])
    
    with tab1:
        st.subheader("Credit Score Trend Analysis")
        
        if date_range == "Last 30 days":
            filtered_data = company_data[company_data['date'] >= datetime.now() - timedelta(days=30)]
        elif date_range == "Last 90 days":
            filtered_data = company_data[company_data['date'] >= datetime.now() - timedelta(days=90)]
        elif date_range == "Last 180 days":
            filtered_data = company_data[company_data['date'] >= datetime.now() - timedelta(days=180)]
        else:
            filtered_data = company_data
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=filtered_data['date'],
            y=filtered_data['credit_score'],
            mode='lines+markers',
            name='Credit Score',
            line=dict(color='#1f77b4', width=3)
        ))
        
        fig.update_layout(
            title=f"Credit Score Trend - {platform.companies[selected_ticker]}",
            xaxis_title="Date",
            yaxis_title="Credit Score",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Score Distribution")
            fig_hist = px.histogram(
                df, 
                x='credit_score', 
                nbins=20,
                title="Overall Credit Score Distribution",
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.subheader("Sector Comparison")
            sector_avg = df.groupby('ticker')['credit_score'].last().reset_index()
            fig_bar = px.bar(
                sector_avg,
                x='ticker',
                y='credit_score',
                title="Current Credit Scores by Company",
                color='credit_score',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.subheader("Feature Contribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="feature-importance">', unsafe_allow_html=True)
            st.subheader("Feature Importance")
            
            fig_importance = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title="Most Influential Credit Factors",
                color='importance',
                color_continuous_scale='Blues'
            )
            fig_importance.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_importance, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("Current Feature Values")
            
            feature_values = []
            for feature in features:
                value = latest_data[feature]
                feature_values.append({
                    'Feature': platform.feature_descriptions.get(feature, feature),
                    'Value': value,
                    'Status': 'Good' if value > np.percentile(df[feature], 60) else 
                             'Poor' if value < np.percentile(df[feature], 40) else 'Average'
                })
            
            feature_df = pd.DataFrame(feature_values)
            
            def color_status(val):
                if val == 'Good':
                    return 'color: green; font-weight: bold'
                elif val == 'Poor':
                    return 'color: red; font-weight: bold'
                else:
                    return 'color: orange; font-weight: bold'
            
            st.dataframe(
                feature_df.style.applymap(color_status, subset=['Status']),
                use_container_width=True
            )
    
    with tab3:
        st.subheader("Model Explainability & Insights")
        
        explanations = platform.generate_explanation(company_data, feature_importance)
        
        st.markdown("### 📋 Credit Assessment Explanation")
        
        for i, explanation in enumerate(explanations, 1):
            st.write(f"{i}. {explanation}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Model Performance")
            model_comparison = pd.DataFrame({
                'Model': ['Decision Tree', 'Random Forest'],
                'Accuracy': [dt_accuracy, rf_accuracy]
            })
            
            fig_models = px.bar(
                model_comparison,
                x='Model',
                y='Accuracy',
                title="Model Accuracy Comparison",
                color='Accuracy',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_models, use_container_width=True)
        
        with col2:
            st.markdown("#### Decision Rules Sample")
            st.info("""
            The model uses interpretable decision rules such as:
            - IF Debt-to-Equity > 1.5 THEN credit penalty
            - IF ROA > 0.1 THEN credit boost  
            - IF Sentiment < 0.3 THEN risk flag
            """)
            
            st.markdown("#### Key Risk Factors")
            risk_factors = [
                "High financial leverage",
                "Declining profitability", 
                "Negative market sentiment",
                "Revenue growth slowdown"
            ]
            
            for factor in risk_factors:
                st.write(f"• {factor}")
    
    with tab4:
        st.subheader("Recent Events & News Impact")
        
        events = platform.simulate_news_events(selected_ticker)
        
        if events:
            for event in events:
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**{event['event']}**")
                        st.write(f"*{event['date'].strftime('%Y-%m-%d')}*")
                    
                    with col2:
                        if event['impact'] == 'positive':
                            st.success("Positive Impact")
                        else:
                            st.error("Negative Impact")
                    
                    st.divider()
        else:
            st.info("No significant recent events detected for this company.")
        
        st.subheader("Event Impact Analysis")
        
        event_impact_data = {
            'Event Type': ['Earnings Beat', 'Debt Issuance', 'Regulatory News', 'M&A Activity'],
            'Avg Score Change': [+15, -8, -12, +10],
            'Frequency': ['Quarterly', 'Occasional', 'Rare', 'Occasional']
        }
        
        event_df = pd.DataFrame(event_impact_data)
        st.dataframe(event_df, use_container_width=True)
    
    st.markdown("---")
    st.markdown(
        "**CredTech Hackathon** | Built with Streamlit | "
        "Data simulated for demonstration purposes"
    )

if __name__ == "__main__":
    main()