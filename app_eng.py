import streamlit as st
import math
from scipy.optimize import fsolve
import numpy as np
import pandas as pd
import plotly.express as px

# Data table with added roughness
tubes_data = {
    "Steel": {
        "roughness": 0.05,
        "D": {
            "1\" 1/4 (DN 32)": 35.9,
            "1\" 1/2 (DN 40)": 41.8,
            "2\" (DN 50)": 53.1,
            "2\" 1/2 (DN 65)": 68.1,
            "3\" (DN 80)": 80.8,
            "3\" 1/2 (DN90)": 93.6,
            "4\" (DN100)": 105.3,
        }
    },
    "Multilayer": {
        "roughness": 0.002,
        "D": {
            "1\" 1/4 (DN 32)": 33,
            "1\" 1/2 (DN 40)": 41,
            "2\" (DN 50)": 51,
            "2\" 1/2 (DN 65)": 58,
            "3\" (DN 80)": 70
        }
    },
    "Copper": {
        "roughness": 0.01,
        "D": {
            "1\" 1/4 (DN 32)": 33,
            "1\" 1/2 (DN 40)": 40,
            "2\" (DN 50)": 50,
            "2\" 1/2 (DN 65)":65,
            "3\" (DN 80)": 80
        }
    }
}

# Static head loss data for each MMTC model
static_head_loss = {
    'MMTC 20': 1.63,
    'MMTC 26': 2.04,
    'MMTC 33': 2.08,
    'MMTC 40': 1.17,
    'MHTC 20': 1.45,
    'MHTC 30': 2.55,
    '2 x MMTC 20': 1.63,
    '2 x MMTC 26': 2.04,
    '2 x MMTC 33': 2.08,
    '2 x MMTC 40': 1.17,
    '2 x MHTC 20': 1.45,
    '2 x MHTC 30': 2.55,
    '3 x MMTC 20': 1.63,
    '3 x MMTC 26': 2.04,
    '3 x MMTC 33': 2.08,
    '3 x MMTC 40': 1.17,
    '3 x MHTC 20': 1.45,
    '3 x MHTC 30': 2.55,
    '4 x MMTC 20': 1.63,
    '4 x MMTC 26': 2.04,
    '4 x MMTC 33': 2.08,
    '4 x MMTC 40': 1.17,
    '4 x MHTC 20': 1.45,
    '4 x MHTC 30': 2.55,
    '5 x MMTC 20': 1.63,
    '5 x MMTC 26': 2.04,
    '5 x MMTC 33': 2.08,
    '5 x MMTC 40': 1.17,
    '5 x MHTC 20': 1.45,
    '5 x MHTC 30': 2.55,
    '6 x MMTC 20': 1.63,
    '6 x MMTC 26': 2.04,
    '6 x MMTC 33': 2.08,
    '6 x MMTC 40': 1.17,
    '6 x MHTC 20': 1.45,
    '6 x MHTC 30': 2.55,
}

# --- Tank data ---
tanks = {
    "B650-B800": {"flow_rate": 4.6, "head_loss": 1.6},
    "B1000": {"flow_rate": 5.1, "head_loss": 2.0},
    "B1500-B3000": {"flow_rate": 5.1, "head_loss": 2.0}
}

# Colebrook-White function
def colebrook(f, epsilon, D, Re):
    return 1 / math.sqrt(f) + 2 * math.log10(epsilon / (3.7 * D) + 2.51 / (Re * math.sqrt(f)))

# Function to calculate head loss per meter
def head_loss_per_meter(f, D, v):
    return f * ((v**2)/2)*(1/D) * 1000/ 10000  # Convert to mmCE/m

# Function to calculate flow velocity
def calculate_velocity(Q, D):
    A = math.pi * (D / 2) ** 2  # Cross-sectional area of the pipe
    v = Q / A
    return v

# Data for pump models
data = {
    'model': [
        'MMTC 20', 'MMTC 26', 'MMTC 33', 'MMTC 40', 
        'MHTC 20', 'MHTC 30', 
        '2 x MMTC 20', '2 x MMTC 26', '2 x MMTC 33', '2 x MMTC 40', 
        '3 x MMTC 20', '3 x MMTC 26', '3 x MMTC 33', '3 x MMTC 40', 
        '2 x MHTC 20', '2 x MHTC 30', '3 x MHTC 20', '3 x MHTC 30', 
        '4 x MMTC 20', '4 x MMTC 26', '4 x MMTC 33', '4 x MMTC 40', 
        '4 x MHTC 20', '4 x MHTC 30', 
        '5 x MMTC 20', '5 x MMTC 26', '5 x MMTC 33', '5 x MMTC 40', 
        '5 x MHTC 20', '5 x MHTC 30', 
        '6 x MMTC 20', '6 x MMTC 26', '6 x MMTC 33', '6 x MMTC 40', 
        '6 x MHTC 20', '6 x MHTC 30'
    ],
    'flow_rate': [
        3.68, 4.72, 5.79, 6.98, 
        3.5, 5.24, 
        2 * 3.68, 2 * 4.72, 2 * 5.79, 2 * 6.98, 
        3 * 3.68, 3 * 4.72, 3 * 5.79, 3 * 6.98, 
        2 * 3.5, 2 * 5.24, 3 * 3.5, 3 * 5.24, 
        4 * 3.68, 4 * 4.72, 4 * 5.79, 4 * 6.98, 
        4 * 3.5, 4 * 5.24, 
        5 * 3.68, 5 * 4.72, 5 * 5.79, 5 * 6.98, 
        5 * 3.5, 5 * 5.24, 
        6 * 3.68, 6 * 4.72, 6 * 5.79, 6 * 6.98, 
        6 * 3.5, 6 * 5.24
    ],
    'available_head': [
        6.3, 3.2, 5.5, 2.8, 
        6.4, 4.4, 
        6.3, 3.2, 5.5, 2.8, 
        6.3, 3.2, 5.5, 2.8, 
        6.4, 4.4, 6.4, 4.4, 
        6.3, 3.2, 5.5, 2.8, 
        6.4, 4.4, 
        6.3, 3.2, 5.5, 2.8, 
        6.4, 4.4, 
        6.3, 3.2, 5.5, 2.8, 
        6.4, 4.4
    ]
}

# Dictionary of temperatures and corresponding kinematic viscosities (in m²/s)
viscosity_data = 1.31e-6

# App title
st.title("Maximum Pipe Length for MMTC and MHTC Heat Pumps")

# Display the kinematic viscosity of water at 10°C
st.write(f"We are using the kinematic viscosity of water at 10°C: **{viscosity_data} m²/s**.")

def main():
    # PAC model selection
    st.markdown('<p style="font-size:20px; margin-bottom: 0px; margin-top: 20px;"><strong>Choose a Heat Pump model:</strong></p>', unsafe_allow_html=True)
    model = st.selectbox("", data['model'])

    # Material selection
    st.markdown('<p style="font-size:20px; margin-bottom: 0px;margin-top: 20px;"><strong>Choose a pipe material:</strong></p>', unsafe_allow_html=True)
    material = st.selectbox("", list(tubes_data.keys()))

    # Pipe size selection based on material
    st.markdown('<p style="font-size:20px; margin-bottom: 0px;margin-top: 20px;"><strong>Choose the pipe size:</strong></p>', unsafe_allow_html=True)
    diameter = st.selectbox("", list(tubes_data[material]["D"].keys()))

    nu = viscosity_data

    # Get the inner diameter and roughness
    Diam = tubes_data[material]["D"][diameter]   # Convert to meters
    D = Diam/1000
    roughness = tubes_data[material]["roughness"]   # Convert to meters
    roughness = roughness/1000
    st.write(f"The inner diameter for the {material} pipe, size {diameter} is: {Diam} mm.")
    st.write(f"Roughness used for the {material} pipe: {roughness} mm.")

    # Find the flow rate corresponding to the chosen model
    index = data['model'].index(model)
    Q_h = data['flow_rate'][index]
    available_head = data['available_head'][index]

    # Convert the volumetric flow rate from m³/h to m³/s
    Q = Q_h / 3600  # 1 hour = 3600 seconds

    # Calculate the flow velocity
    v = calculate_velocity(Q, D)
    if v > 1.5:
        st.markdown(f"<div style='background-color: red; padding: 10px;'>The flow velocity is {v:.2f} m/s</div>", unsafe_allow_html=True)
    else:
        st.write(f"Flow velocity: {v:.2f} m/s")

    # Calculate the Reynolds number with the new value of nu
    Re = (v * D) / nu
    st.write(f"Reynolds: {Re:.0f}")

    # Number of 90° wide-angle elbows
    st.markdown('<p style="font-size:20px; margin-bottom: 0px;margin-top: 20px;"><strong>Number of 90° wide-angle elbows:</strong></p>', unsafe_allow_html=True)
    elbows = st.selectbox("", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 , 17, 18, 19, 20])

    # Additional head losses (balancing valve, coil tank, accidents,...(mCE))
    st.markdown('<p style="font-size:20px; margin-bottom: 0px;margin-top: 20px;"><strong>Additional head losses (balancing valve, tank exchanger, elbows,...) in mCE</strong></p>', unsafe_allow_html=True)

    other_loss1 = st.number_input(".", min_value=0.0, step=0.1, key='other_loss1')
    other_loss2 = st.number_input(".", min_value=0.0, step=0.1, key='other_loss2')
    other_loss3 = st.number_input(".", min_value=0.0, step=0.1, key='other_loss3')

    other_loss = other_loss1 + other_loss2 + other_loss3

    # Choice to deduct the static head loss
    st.markdown('<p style="font-size:20px; margin-bottom: 0px;margin-top: 20px;"><strong>Deduction of the total theoretical head loss for each Heat Pump: Isolation Valve, Filter, Check Valve, T and Buffer Tank Flanges depending on the model:</strong></p>', unsafe_allow_html=True)
    deduct_static_loss = st.checkbox("Deduct", value=True)

    total_static_loss = 0

    # v²/2g
    y = (v**2)/(2*9.81)

    # Head loss for 90° elbows
    elbow_loss = y * elbows * 0.45

    # If deducting the static head loss
    if deduct_static_loss and model in static_head_loss:
        static_loss = static_head_loss[model]
        total_static_loss = elbow_loss + static_loss + other_loss1 + other_loss2 + other_loss3
        st.write(f"Deducted static head loss: {static_loss} mCE")
        st.image("Tableau_pac_1.png")
    else:
        total_static_loss = elbow_loss + other_loss

    # Calculate the head loss per meter and the possible pipe length
    if Re > 2000:
        initial_guess = 0.02
        f_solution, = fsolve(colebrook, initial_guess, args=(roughness, D, Re))
        loss_per_meter = head_loss_per_meter(f_solution, D, v)
        possible_length = (available_head - total_static_loss ) / loss_per_meter

        # Display the results
        st.write(f"The friction coefficient is {f_solution:.3f}")
        st.write(f"The total singular head losses are {total_static_loss:.3f} mCE")
        st.write(f"The head loss per meter is {loss_per_meter:.3f} mCE/m")

        if possible_length > 0:
            st.markdown(f"<div style='background-color: lightgreen; padding: 10px;'>The maximum Pipe run back and forth length is {possible_length:.2f} meters</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: red; padding: 10px;'>The head losses exceed the available head, no length is possible.</div>", unsafe_allow_html=True)
    else:
        st.write("The flow is laminar, use another calculation method.")

    st.write("")
    st.write("")
    st.divider()
    st.write("")

    # --- User interface ---
    st.title("Calculation of B Tanks Head Losses at Heat Pump Flow Rates")
    st.markdown("Enter the flow rate and choose the tank to display the head losses and the corresponding graph.")

    st.image('Tableau_pac_2.jpg')

    tank_flow_rate = st.slider(
        "Choose a Tank primary flow rate :",
        min_value = 0.0,
        max_value= 10.0,
        step=0.1,
        value=5.  # Default value
    )

    # 2️⃣ Tank selection
    chosen_tank = st.selectbox(
        "Choose the type of tank:", options=list(tanks.keys())
    )

    # --- Retrieve the chosen tank data ---
    reference_flow_rate = tanks[chosen_tank]["flow_rate"]
    reference_head_loss = tanks[chosen_tank]["head_loss"]

    # --- Calculate head losses (quadratic head loss law) ---
    def calculate_head_loss(flow_rate, reference_flow_rate, reference_head_loss):
        """Calculate the head loss for a given flow rate using the quadratic law."""
        return reference_head_loss * (flow_rate / reference_flow_rate) ** 2

    user_head_loss = calculate_head_loss(tank_flow_rate, reference_flow_rate, reference_head_loss)

    # 3️⃣ Display the results
    st.subheader("Results")
    st.markdown(f"**Chosen tank** : {chosen_tank}")
    st.markdown(f"**Entered flow rate** : {tank_flow_rate:.2f} m³/h")
    st.markdown(f"**Head losses** : {user_head_loss:.2f} mCE")

    # 4️⃣ Plot the graph
    flow_rates = np.linspace(0, 10, 100)  # Flow rate range from 0 to 10 m³/h
    head_loss_values = [calculate_head_loss(d, reference_flow_rate, reference_head_loss) for d in flow_rates]  # Calculate head losses

    # Create a DataFrame for plotting the graph
    df = pd.DataFrame({
        "Flow Rate (m³/h)": flow_rates,
        "Head Loss (mCE)": head_loss_values
    })

    # Plot with the operating point
    fig = px.line(df, x="Flow Rate (m³/h)", y="Head Loss (mCE)", title="Head Loss Curve")
    fig.add_scatter(
        x=[tank_flow_rate], 
        y=[user_head_loss], 
        mode='markers+text', 
        marker=dict(size=10, color='red'), 
        name='Operating Point',
        text=["Operating Point"],
        textposition="bottom right"
    )

    # Plot dashed lines
    fig.add_scatter(  
        x=[tank_flow_rate, tank_flow_rate], 
        y=[0, user_head_loss], 
        mode='lines', 
        line=dict(dash='dot', color='red'), 
        name='Dashed Line'
    )
    
    fig.add_scatter(
        x=[0, tank_flow_rate], 
        y=[user_head_loss, user_head_loss], 
        mode='lines', 
        line=dict(dash='dot', color='red'), 
        name='Dashed Line'
    )

    # Display the graph
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()





