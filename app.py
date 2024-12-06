from flask import Flask, render_template, request, jsonify
from transformers import pipeline

# Inicializa la aplicación Flask
app = Flask(__name__)

# Cargar el modelo de preguntas y respuestas
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Contexto de la biblioteca
context = """
La biblioteca de la universidad está ubicada en el centro tutorial de san juan nepomuceno.
La biblioteca está abierta de lunes a viernes de 6:00 AM a 8:00 PM, y los sábados de 8:00 AM a 5:00 PM.
Puedes sacar libros presentando tu tarjeta de biblioteca, que puedes obtener en la oficina de servicios al estudiante.
Tenemos una gran colección de libros sobre ciencia, tecnología, literatura y más.
Nuestros libros mas populares son: Don quijote de la mancha, Cien años de soledad y Rafael pombo. 
"""

# Función para responder preguntas
def chatbot_response_ia(question):  
    response = qa_pipeline({
        'context': context,
        'question': question
    })
    return response['answer']

# Ruta principal (inicio)
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para procesar la pregunta del usuario
@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.form['message']
    response = chatbot_response_ia(user_message)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
