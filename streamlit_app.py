import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Superplasticizer Dosage Predictor",
    page_icon="🧪",
    layout="wide"
)

# Load models
@st.cache_resource
def load_models():
    """Load trained model and encoders"""
    model = joblib.load('optimal_dosage_predictor.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    saturation_data = pd.read_csv('saturation_points.csv')
    flow_data = pd.read_csv('processed_flow_data.csv')
    return model, label_encoder, saturation_data, flow_data

try:
    model, le, saturation_df, flow_df = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"⚠️ Error loading models: {e}")

# Title and description
st.title("🧪 Optimal Superplasticizer Dosage Predictor")
st.markdown("""
This tool predicts the optimal dosage of superplasticizer (PCE or SNF) for cement paste 
based on the **Marsh Cone Test** results. The optimal dosage represents the **saturation point** 
where adding more superplasticizer does not improve fluidity.
""")

st.divider()

if models_loaded:
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📋 Input Parameters")
        
        # User inputs
        wc_ratio = st.slider(
            "Water-to-Cement Ratio (W/C)",
            min_value=0.30,
            max_value=0.50,
            value=0.40,
            step=0.01,
            help="Typical range: 0.35 - 0.45 for high-strength concrete"
        )
        
        sp_type = st.selectbox(
            "Superplasticizer Type",
            options=['PCE', 'SNF'],
            help="PCE = Polycarboxylate Ether, SNF = Sulfonated Naphthalene Formaldehyde"
        )
        
        # Silica Fume (placeholder for future enhancement)
        st.info("💡 **Note:** Current model trained for plain cement paste (0% Silica Fume). Future versions will include SF variations.")
        
        # Predict button
        if st.button("🔮 Predict Optimal Dosage", type="primary"):
            # Encode and predict
            sp_encoded = le.transform([sp_type])[0]
            X = np.array([[wc_ratio, sp_encoded]])
            prediction = model.predict(X)[0]
            
            # Store in session state
            st.session_state['prediction'] = prediction
            st.session_state['wc_ratio'] = wc_ratio
            st.session_state['sp_type'] = sp_type
    
    with col2:
        st.subheader("🎯 Prediction Results")
        
        if 'prediction' in st.session_state:
            pred = st.session_state['prediction']
            wc = st.session_state['wc_ratio']
            sp = st.session_state['sp_type']
            
            # Display prediction in a nice metric card
            st.metric(
                label=f"Optimal {sp} Dosage",
                value=f"{pred:.3f}%",
                help="This is the saturation point dosage"
            )
            
            # Additional information
            st.success("✅ Prediction Complete!")
            
            with st.expander("📊 Interpretation & Guidelines", expanded=True):
                st.markdown(f"""
                **Recommended Dosage:** `{pred:.3f}%` of cement weight
                
                **What this means:**
                - For 100 kg of cement, use **{pred*1000:.1f} grams** of {sp}
                - This is the **saturation point** for W/C ratio {wc:.2f}
                - Adding more than this amount will **not improve** fluidity
                - May cause segregation or bleeding if exceeded
                
                **Practical Application:**
                - Use this dosage for your concrete mix design
                - Adjust slightly based on field conditions
                - Monitor slump flow and workability
                """)
            
            # Show comparison with training data
            st.markdown("### 📈 Comparison with Training Data")
            
            # Find closest actual values
            actual_data = saturation_df[saturation_df['sp_type'] == sp].copy()
            actual_data['wc_diff'] = abs(actual_data['wc_ratio'] - wc)
            closest = actual_data.nsmallest(2, 'wc_diff')
            
            if not closest.empty:
                comparison_df = pd.DataFrame({
                    'W/C Ratio': [wc] + closest['wc_ratio'].tolist(),
                    'Dosage (%)': [pred] + closest['optimal_dosage'].tolist(),
                    'Type': ['Predicted'] + ['Actual']*len(closest)
                })
                
                fig = px.bar(
                    comparison_df,
                    x='W/C Ratio',
                    y='Dosage (%)',
                    color='Type',
                    barmode='group',
                    title=f'{sp} Dosage Comparison',
                    color_discrete_map={'Predicted': '#FF6B6B', 'Actual': '#4ECDC4'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Enter parameters and click 'Predict' to see results")
    
    # Visualizations section
    st.divider()
    st.subheader("📊 Flow Curve Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Flow Curves", "Saturation Points", "Data Table"])
    
    with tab1:
        st.markdown("**Flow time vs. Superplasticizer dosage for different W/C ratios**")
        
        selected_sp = st.radio("Select Superplasticizer", ['PCE', 'SNF'], horizontal=True, key='flow_curve_sp')
        
        # Filter data
        filtered_data = flow_df[flow_df['sp_type'] == selected_sp]
        
        # Create plot
        fig = go.Figure()
        
        colors = {0.35: '#E63946', 0.40: '#F77F00', 0.45: '#06A77D'}
        
        for wc in [0.35, 0.40, 0.45]:
            subset = filtered_data[filtered_data['wc_ratio'] == wc]
            fig.add_trace(go.Scatter(
                x=subset['dosage'],
                y=subset['flow_time'],
                mode='lines+markers',
                name=f'W/C {wc}',
                line=dict(width=3, color=colors[wc]),
                marker=dict(size=10)
            ))
            
            # Add saturation point marker
            sat_point = saturation_df[(saturation_df['sp_type'] == selected_sp) & 
                                     (saturation_df['wc_ratio'] == wc)]
            if not sat_point.empty:
                fig.add_trace(go.Scatter(
                    x=[sat_point['optimal_dosage'].values[0]],
                    y=[sat_point['min_flow_time'].values[0]],
                    mode='markers',
                    name=f'Saturation (W/C {wc})',
                    marker=dict(size=15, symbol='star', color=colors[wc], 
                              line=dict(width=2, color='black')),
                    showlegend=False
                ))
        
        fig.update_layout(
            title=f'{selected_sp} Flow Curves (⭐ = Saturation Point)',
            xaxis_title='Dosage (%)',
            yaxis_title='Flow Time (seconds)',
            hovermode='x unified',
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("**Comparison of saturation dosages between PCE and SNF**")
        
        # Saturation comparison
        fig = px.bar(
            saturation_df,
            x='wc_ratio',
            y='optimal_dosage',
            color='sp_type',
            barmode='group',
            title='Saturation Dosage Comparison',
            labels={'wc_ratio': 'W/C Ratio', 'optimal_dosage': 'Optimal Dosage (%)', 'sp_type': 'Superplasticizer'},
            color_discrete_map={'PCE': '#2E86AB', 'SNF': '#A23B72'},
            text='optimal_dosage'
        )
        
        fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
        fig.update_layout(height=500)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**PCE Summary**")
            pce_stats = saturation_df[saturation_df['sp_type'] == 'PCE']['optimal_dosage']
            st.metric("Average Dosage", f"{pce_stats.mean():.3f}%")
            st.metric("Range", f"{pce_stats.min():.2f}% - {pce_stats.max():.2f}%")
        
        with col2:
            st.markdown("**SNF Summary**")
            snf_stats = saturation_df[saturation_df['sp_type'] == 'SNF']['optimal_dosage']
            st.metric("Average Dosage", f"{snf_stats.mean():.3f}%")
            st.metric("Range", f"{snf_stats.min():.2f}% - {snf_stats.max():.2f}%")
    
    with tab3:
        st.markdown("**Complete saturation points dataset**")
        st.dataframe(
            saturation_df.style.format({
                'wc_ratio': '{:.2f}',
                'optimal_dosage': '{:.3f}',
                'min_flow_time': '{:.1f}'
            }),
            use_container_width=True
        )
        
        # Download button
        csv = saturation_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data (CSV)",
            data=csv,
            file_name="saturation_points.csv",
            mime="text/csv"
        )

    # Model information
    st.divider()
    with st.expander("ℹ️ About the Model"):
        st.markdown("""
        **Model Details:**
        - **Algorithm:** Gradient Boosting Regressor
        - **Training Data:** 6 saturation points (3 W/C ratios × 2 SP types)
        - **Performance:** MAE = 0.025%, RMSE = 0.035%
        - **Validation:** Leave-One-Out Cross-Validation
        
        **Input Features:**
        1. Water-to-Cement Ratio (0.35 - 0.45)
        2. Superplasticizer Type (PCE or SNF)
        
        **Output:**
        - Optimal dosage at saturation point (%)
        
        **Limitations:**
        - Trained on plain cement paste (no Silica Fume)
        - Best for W/C ratios between 0.35 and 0.45
        - Results may vary with different cement types
        - Field testing recommended for final validation
        """)

else:
    st.warning("Please ensure all model files are present in the working directory.")
