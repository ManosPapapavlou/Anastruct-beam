import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI plotting
from flask import Flask, render_template, request, redirect, url_for
from anastruct import SystemElements
import io
import base64
import matplotlib.pyplot as plt
import pandas as pd  # Import pandas for DataFrame handling

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        num_spans = int(request.form['num_spans'])
        width = float(request.form['width'])
        height = float(request.form['height'])
        return redirect(url_for('input_properties', num_spans=num_spans, width=width, height=height))

    return render_template('index.html')

@app.route('/input-properties', methods=['GET', 'POST'])
def input_properties():
    if request.method == 'POST':
        num_spans = int(request.form['num_spans'])
        width = float(request.form['width'])
        height = float(request.form['height'])

        # Debugging: Print the entire form data
        print("Received form data:")
        print(request.form)

        # Get element data from form
        element_data = request.form.getlist('element_data')
        print(f"Received element data: {element_data}")  # Debug print

        # Check if element_data has the expected number of values
        expected_element_data_length = num_spans * 2  # Each element has a length and a load
        if len(element_data) != expected_element_data_length:
            raise ValueError(f"Expected {expected_element_data_length} values for element data, but got {len(element_data)}")

        # Parse the element data
        elements = [{'length': float(element_data[i * 2]), 'load': float(element_data[i * 2 + 1])} for i in range(num_spans)]

        # Get node data from form
        node_data = request.form.getlist('node_data_support')
        node_loads_vert = request.form.getlist('node_load_vert')
        node_loads_horiz = request.form.getlist('node_load_horiz')

        print(f"Received node data: {node_data}")  # Debug print
        print(f"Received node vertical loads: {node_loads_vert}")  # Debug print
        print(f"Received node horizontal loads: {node_loads_horiz}")  # Debug print

        # Ensure node data is complete
        if len(node_data) != num_spans + 1:
            raise ValueError(f"Expected {num_spans + 1} values for node support types, but got {len(node_data)}")

        if len(node_loads_vert) != num_spans + 1 or len(node_loads_horiz) != num_spans + 1:
            raise ValueError(f"Expected {num_spans + 1} values for node loads, but got {len(node_loads_vert)} vertical loads and {len(node_loads_horiz)} horizontal loads")

        # Create node list
        nodes = [{'support_type': node_data[i], 'Fy': float(node_loads_vert[i]), 'Fx': float(node_loads_horiz[i])} for i in range(num_spans + 1)]

        # Set material properties
        A = width * height
        I = width * (height ** 3) / 12
        E = 33000000
        ss = SystemElements(EA=E * A, EI=E * I)

        # Add elements to the system
        x_start = 0.0
        for element in elements:
            x_end = x_start + element['length']
            ss.add_element(location=[[x_start, 0], [x_end, 0]])
            x_start = x_end

        # Add supports and node loads to the system
        for i, node in enumerate(nodes):
            node_id = i + 1
            # Add support types
            if node['support_type'] == 'fixed':
                ss.add_support_fixed(node_id=node_id)
            elif node['support_type'] == 'hinged':
                ss.add_support_hinged(node_id=node_id)
            elif node['support_type'] == 'roller':
                ss.add_support_roll(node_id=node_id, direction=2)

            # Add node loads (vertical Fy and horizontal Fx)
            if node['Fy'] != 0:
                ss.point_load(node_id=node_id, Fy=node['Fy'])
            if node['Fx'] != 0:
                ss.point_load(node_id=node_id, Fx=node['Fx'])

        # Add distributed loads to elements
        for i, element in enumerate(elements):
            ss.q_load(element_id=i + 1, q=element['load'])

        # Solve the system
        ss.solve()

        # Extract reaction forces from node results
        node_results = ss.get_node_results_system()

        # Convert node results to DataFrame and print
        df_reactions = pd.DataFrame(node_results)
        print("Reaction Forces:")
        print(df_reactions)

        # Extract moments for each element
        element_results = ss.get_element_results()

        # Convert element results to DataFrame and print
        df_elements = pd.DataFrame(element_results)
        print("Element Results (Moments, Shear Forces, etc.):")
        print(df_elements)

        # Extract displacements for elements (umax, umin, etc.)
        element_displacements = []
        for element in element_results:
            element_displacements.append({
                'id': element['id'],
                'wmax': element['wmax'] * 1000,  # Maximum displacement for the element
                'wmin': element['wmin'] * 1000,  # Minimum displacement for the element
            })

        print("Element Displacements:")
        print(element_displacements)

        # Plot structure and save as image
        img = io.BytesIO()
        ss.show_structure()
        plt.savefig(img, format='png')
        img.seek(0)
        structure_url = base64.b64encode(img.getvalue()).decode()

        # Plot bending moment diagram
        img = io.BytesIO()
        ss.show_bending_moment()
        plt.savefig(img, format='png')
        img.seek(0)
        moment_url = base64.b64encode(img.getvalue()).decode()

        # Plot shear force diagram
        img = io.BytesIO()
        ss.show_shear_force()
        plt.savefig(img, format='png')
        img.seek(0)
        shear_url = base64.b64encode(img.getvalue()).decode()

        # Plot displacement diagram
        img = io.BytesIO()
        ss.show_displacement()
        plt.savefig(img, format='png')
        img.seek(0)
        displacement_url = base64.b64encode(img.getvalue()).decode()

        # Render the results to the results.html
        return render_template(
            'results.html',
            width=width,
            height=height,
            reactions=df_reactions.to_dict(orient='records'),
            moments=df_elements.to_dict(orient='records'),
            shear_forces=df_elements.to_dict(orient='records'),
            displacements=df_elements.to_dict(orient='records'),
            structure_url=structure_url,
            moment_url=moment_url,
            shear_url=shear_url,
            displacement_url=displacement_url
        )
    
    num_spans = int(request.args['num_spans'])
    width = float(request.args['width'])
    height = float(request.args['height'])

    return render_template('input_properties.html', num_spans=num_spans, width=width, height=height)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
