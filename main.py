import numpy as np
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.datasets import mnist

class SOM3D:
    def __init__(self, x, y, z, input_dim, learning_rate=0.1, radius=None, sigma_decay=0.1):
        self.x = x
        self.y = y
        self.z = z
        self.input_dim = input_dim
        self.learning_rate_initial = learning_rate
        self.learning_rate = learning_rate
        self.radius_initial = radius if radius else max(x, y, z) / 2
        self.radius = self.radius_initial
        self.sigma_decay = sigma_decay
        # Initialize weights randomly
        self.weights = np.random.random((x, y, z, input_dim))
        # Create neuron coordinate grid
        self.neuron_coords = np.array(
            np.meshgrid(range(x), range(y), range(z), indexing='ij')
        ).reshape(3, -1).T  # Shape: (x*y*z, 3)
    
    def find_bmu(self, input_vector):
        # Compute Euclidean distance between input_vector and all neurons
        # Efficient vectorized computation
        diff = self.weights - input_vector  # Shape: (x, y, z, input_dim)
        distances = np.linalg.norm(diff, axis=3)  # Shape: (x, y, z)
        bmu_index = np.unravel_index(np.argmin(distances), distances.shape)
        return np.array(bmu_index)  # Convert to numpy array for vector operations
    
    def update_weights(self, input_vector, bmu, iteration, total_iterations):
        # Decay learning rate and radius
        self.learning_rate = self.learning_rate_initial * np.exp(-iteration / total_iterations)
        self.radius = self.radius_initial * np.exp(-iteration / total_iterations)
        
        # Compute the influence of each neuron
        # Get all neuron coordinates
        neuron_coords = self.neuron_coords  # Shape: (x*y*z, 3)
        
        # Compute distances from BMU to all neurons
        distances = np.linalg.norm(neuron_coords - bmu, axis=1)  # Shape: (x*y*z,)
        
        # Compute the neighborhood function (Gaussian)
        influence = np.exp(-(distances ** 2) / (2 * (self.radius ** 2)))  # Shape: (x*y*z,)
        
        # Find neurons within the current radius
        mask = influence > 0
        affected_neurons = np.where(mask)[0]
        
        # Update the weights for affected neurons
        if len(affected_neurons) > 0:
            # Reshape weights to (x*y*z, input_dim)
            weights_reshaped = self.weights.reshape(-1, self.input_dim)
            # Compute the influence scaling
            influence_scaling = influence[affected_neurons].reshape(-1, 1)  # Shape: (num_affected, 1)
            # Compute the weight updates
            delta = self.learning_rate * influence_scaling * (input_vector - weights_reshaped[affected_neurons])
            # Update weights
            weights_reshaped[affected_neurons] += delta
            # Reshape back to original
            self.weights = weights_reshaped.reshape(self.x, self.y, self.z, self.input_dim)
    
    def calculate_quantization_error(self, data):
        error = 0
        for input_vector in data:
            bmu = self.find_bmu(input_vector)
            error += np.linalg.norm(input_vector - self.weights[tuple(bmu)])
        return error / len(data)
    
    def add_neuron(self, bmu):
        # Example: Add a neuron in the dimension with the smallest size
        dims = [self.x, self.y, self.z]
        min_dim = np.argmin(dims)
        if min_dim == 0:
            self.x += 1
        elif min_dim == 1:
            self.y += 1
        else:
            self.z += 1
        # Pad the weights array accordingly
        pad_width = [(0,0), (0,0), (0,0), (0,0)]
        pad_width[min_dim] = (0,1)
        self.weights = np.pad(self.weights, pad_width, mode='constant', constant_values=0)
        # Initialize new neuron's weights randomly
        index = [slice(None)] * 4
        index[min_dim] = -1
        if min_dim == 0:
            # Assign to self.weights[-1, :, :, :] which has shape (y, z, input_dim)
            self.weights[tuple(index)] = np.random.random((self.y, self.z, self.input_dim))
        elif min_dim == 1:
            # Assign to self.weights[:, -1, :, :] which has shape (x, z, input_dim)
            self.weights[tuple(index)] = np.random.random((self.x, self.z, self.input_dim))
        else:
            # Assign to self.weights[:, :, -1, :] which has shape (x, y, input_dim)
            self.weights[tuple(index)] = np.random.random((self.x, self.y, self.input_dim))
        print(f"Added a neuron along dimension {['X', 'Y', 'Z'][min_dim]}. New grid size: ({self.x}, {self.y}, {self.z})")
    
    def remove_neuron(self, bmu):
        # Example: Remove a neuron from the dimension with the largest size
        dims = [self.x, self.y, self.z]
        max_dim = np.argmax(dims)
        if dims[max_dim] > 1:
            if max_dim == 0:
                self.weights = self.weights[:-1, :, :, :]
                self.x -= 1
            elif max_dim == 1:
                self.weights = self.weights[:, :-1, :, :]
                self.y -= 1
            else:
                self.weights = self.weights[:, :, :-1, :]
                self.z -= 1
            print(f"Removed a neuron from dimension {['X', 'Y', 'Z'][max_dim]}. New grid size: ({self.x}, {self.y}, {self.z})")
        else:
            print(f"Cannot remove neuron from dimension {['X', 'Y', 'Z'][max_dim]} as its size is already 1.")
    
    def train_dynamic(self, data, num_iterations, qe_threshold_add, qe_threshold_remove):
        for iteration in range(num_iterations):
            input_vector = data[np.random.randint(0, len(data))]
            bmu = self.find_bmu(input_vector)
            self.update_weights(input_vector, bmu, iteration, num_iterations)
            
            # Periodically evaluate QE and adapt
            if iteration % 1000 == 0 and iteration != 0:
                qe = self.calculate_quantization_error(data)
                st.write(f"Iteration {iteration}/{num_iterations}, QE: {qe:.4f}")
                if qe > qe_threshold_add:
                    st.write("Adding a neuron due to high QE")
                    self.add_neuron(bmu)
                elif qe < qe_threshold_remove:
                    st.write("Removing a neuron due to low QE")
                    self.remove_neuron(bmu)
    
    def visualize_som(self):
        x_coords, y_coords, z_coords = np.indices((self.x, self.y, self.z))
        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        z_flat = z_coords.flatten()
        
        # For visualization purposes, map weight vectors to colors
        # Since weight vectors are high-dimensional, use PCA to reduce to 3D for RGB
        pca = PCA(n_components=3)
        weights_reshaped = self.weights.reshape(-1, self.input_dim)
        weights_pca = pca.fit_transform(weights_reshaped)
        # Normalize to [0,1] for RGB
        weights_pca -= weights_pca.min(axis=0)
        weights_pca /= weights_pca.max(axis=0)
        colors = ['rgb({},{},{})'.format(int(w[0]*255), int(w[1]*255), int(w[2]*255)) for w in weights_pca]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x_flat,
            y=y_flat,
            z=z_flat,
            mode='markers',
            marker=dict(
                size=3,
                color=colors,
                opacity=0.8
            )
        )])
        fig.update_layout(scene=dict(
                            xaxis_title='X',
                            yaxis_title='Y',
                            zaxis_title='Z'),
                          title="Dynamic 3D Self-Organizing Map")
        st.plotly_chart(fig)

def main():
    st.title("Dynamic 3D Self-Organizing Map on MNIST")
    
    st.sidebar.header("SOM Configuration")
    
    # Hyperparameters
    x = st.sidebar.slider("Grid X", 5, 20, 10)
    y = st.sidebar.slider("Grid Y", 5, 20, 10)
    z = st.sidebar.slider("Grid Z", 5, 20, 10)
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
    num_iterations = st.sidebar.number_input("Iterations", min_value=1000, max_value=100000, value=10000, step=1000)
    qe_threshold_add = st.sidebar.slider("QE Threshold Add", 0.0, 2.0, 1.0)
    qe_threshold_remove = st.sidebar.slider("QE Threshold Remove", 0.0, 2.0, 0.2)
    apply_pca = st.sidebar.checkbox("Apply PCA for Dimensionality Reduction", value=True)
    pca_components = st.sidebar.slider("PCA Components", 10, 100, 50) if apply_pca else 784
    
    # Load MNIST data
    @st.cache(allow_output_mutation=True)
    def load_data():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        data = np.concatenate((x_train, x_test), axis=0)
        labels = np.concatenate((y_train, y_test), axis=0)
        data = data.reshape((data.shape[0], -1))  # Flatten
        scaler = MinMaxScaler()
        data_normalized = scaler.fit_transform(data)
        if apply_pca:
            pca = PCA(n_components=pca_components)
            data_reduced = pca.fit_transform(data_normalized)
            return data_reduced, labels
        else:
            return data_normalized, labels
    
    data, labels = load_data()
    st.write(f"Data Loaded: {data.shape[0]} samples with {data.shape[1]} features each.")
    
    if st.button("Start Training"):
        # Initialize SOM
        som = SOM3D(x, y, z, data.shape[1], learning_rate)
        
        # Train SOM with dynamic adaptation
        with st.spinner("Training SOM..."):
            som.train_dynamic(data, num_iterations, qe_threshold_add, qe_threshold_remove)
        
        st.success("Training Completed")
        
        # Visualize the SOM
        with st.spinner("Visualizing SOM..."):
            som.visualize_som()
        
        # Optional: Visualize data labels mapped to BMUs
        visualize_labels_on_som(som, data, labels)

def visualize_labels_on_som(som, data, labels):
    st.header("Data Labels on SOM")
    # Assign labels to BMUs
    label_map = {}
    for i in range(len(data)):
        input_vector = data[i]
        bmu = som.find_bmu(input_vector)
        bmu_key = tuple(bmu)
        if bmu_key in label_map:
            label_map[bmu_key].append(labels[i])
        else:
            label_map[bmu_key] = [labels[i]]
    
    # Determine the most common label for each BMU
    bmu_labels = {}
    for bmu, lbls in label_map.items():
        most_common = max(set(lbls), key=lbls.count)
        bmu_labels[bmu] = most_common
    
    # Prepare data for visualization
    x_coords, y_coords, z_coords = np.indices((som.x, som.y, som.z))
    x_flat = x_coords.flatten()
    y_flat = y_coords.flatten()
    z_flat = z_coords.flatten()
    
    # Assign colors based on labels
    label_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF',
                   '#00FFFF', '#800000', '#808000', '#008080', '#800080']  # Colors for digits 0-9
    colors = []
    for i in range(som.x):
        for j in range(som.y):
            for k in range(som.z):
                label = bmu_labels.get((i,j,k), -1)
                if label == -1:
                    colors.append('rgb(200,200,200)')  # Gray for unassigned
                else:
                    colors.append(label_colors[label])
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x_flat,
        y=y_flat,
        z=z_flat,
        mode='markers',
        marker=dict(
            size=5,
            color=colors,
            opacity=0.8
        )
    )])
    fig.update_layout(scene=dict(
                        xaxis_title='X',
                        yaxis_title='Y',
                        zaxis_title='Z'),
                      title="MNIST Labels Mapped on SOM")
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
