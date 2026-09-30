import streamlit as st
import matplotlib.pyplot as plt
from nn_core import DATASETS, make_data, build_model, train, plot_state

st.set_page_config(page_title="Neural Network Playground", page_icon="🧠", layout="wide")
st.title("🧠 Interactive Neural Network Learning Demo")
st.caption("Built with TensorFlow/Keras + Streamlit  |  Kirish  |  RA2411056050002  |  B.Tech CSE-DS")

with st.sidebar:
    st.header("Controls")
    ds = st.selectbox("Dataset", DATASETS)
    noise = st.slider("Noise", 0.0, 0.5, 0.2, 0.05)
    layers = st.slider("Hidden layers", 1, 5, 2)
    neurons = st.slider("Neurons per layer", 1, 16, 8)
    act = st.selectbox("Activation", ["relu", "tanh", "sigmoid", "linear"])
    lr = st.select_slider("Learning rate", [0.001, 0.003, 0.01, 0.03, 0.1], value=0.03)
    epochs = st.slider("Epochs", 10, 300, 100, 10)
    go = st.button("▶ Train network", type="primary", use_container_width=True)

tab1, tab2 = st.tabs(["Playground", "How it works"])
with tab1:
    Xtr, Xte, ytr, yte = make_data(ds, noise)
    model = build_model(layers, neurons, act, lr)
    c1, c2, c3 = st.columns(3)
    c1.metric("Trainable parameters", f"{model.count_params():,}")
    c2.metric("Architecture", "2 → " + " → ".join([str(neurons)] * layers) + " → 1")
    slot, stat = st.empty(), c3.empty()
    if go:
        bar = st.progress(0)
        def upd(ep, h):
            fig = plot_state(model, Xtr, ytr, Xte, yte, h)
            slot.pyplot(fig); plt.close(fig)
            bar.progress(ep / epochs)
            stat.metric("Test accuracy", f"{h['val_accuracy'][-1]*100:.1f}%")
        with st.spinner("Training..."):
            train(model, Xtr, ytr, Xte, yte, epochs, upd, every=max(1, epochs // 15))
        st.success("Training complete. Change the settings and train again to compare.")
    else:
        st.info("Pick a dataset and settings in the sidebar, then press **Train network**.")
        fig = plot_state(model, Xtr, ytr, Xte, yte, {"loss": [], "val_loss": []})
        slot.pyplot(fig); plt.close(fig)
    st.markdown("**Try:** Spiral with 1 layer / 2 neurons underfits. Add layers and neurons and watch the boundary fit. A linear activation cannot solve Moons.")
with tab2:
    st.markdown("""
**Neuron:** output = activation(w·x + b). **Layers** stack neurons so the network can bend the decision boundary.
**Training:** forward pass → binary cross-entropy loss → backpropagation → Adam updates weights.
**Activation:** non-linear functions (ReLU, tanh, sigmoid) let networks learn curved boundaries; *linear* collapses to a straight line.
**Learning rate:** too high makes loss jump around, too low trains slowly.
**Train vs test loss:** a growing gap between them signals overfitting.
Background colour = predicted probability of class 1 (blue) vs class 0 (red); circles are training points, squares are test points.
""")
