
from flask import Flask, request, jsonify, render_template
from mobilelegend.backoffmodel import BackoffNGramModel
from mobilelegend.interpolationmodel import InterpolationModel
from mobilelegend.preload import process_text_files_in_folder


# get token
train_token, val_token, test_token = process_text_files_in_folder('./corpus')

backoff_model = BackoffNGramModel(max_n=4)
backoff_model.build_model(train_token)


inter_model = InterpolationModel(max_n=4,lambdas=[0.1, 0.2, 0.3, 0.4])
inter_model.build_model(train_token)


generated_word = ''
app = Flask(__name__, template_folder='templates')

@app.route('/')
def index():
    return render_template('./index.html')  # Render the main HTML page



@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    context = data.get('context', '').split()  # Ensure context is a list of words
    model_type = data.get('model', 'backoff')  # Default to backoff if model not specified
    words = data.get('word', 30)
    
    if model_type == 'backoff':
        context = tuple(context[-(backoff_model.max_n - 1):])
        # Generate text
        generated_text = backoff_model.generate_text(seed=context, length=int(words))
        
         
    elif model_type == 'interpolation':
        generated_text = inter_model.generate_text(seed=context, length=int(words))
        
    else:
        return jsonify({'generated_text': 'Something went wrong try again'}), 400
    
    return jsonify({'generated_text': generated_text}), 200











if __name__ == '__main__':
    app.run(debug=True)
