from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain.llms import OpenAI
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_sql_agent
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from sqlalchemy.engine import URL
from sqlalchemy import create_engine, text
import os
import gradio as gr
import re
import matplotlib.pyplot as plt
os.environ['OPENAI_API_KEY'] = "sk-**"

db_user = "gpt_reader"
db_password = ""
db_host = ""
db_name = "t"

connection_string = 'DRIVER={ODBC Driver 18 for SQL Server};SERVER=' + db_host + ';DATABASE=' + db_name + ';UID=' + db_user + ';PWD=' + db_password
connection_url = str(URL.create("mssql+pyodbc", query={"odbc_connect": connection_string}))
db = SQLDatabase.from_uri(connection_url)

# connect to db using sqllite3 aswell
engine = create_engine(connection_url, echo=True)
connection = engine.connect()



agent_outputs = []


def new_agent_action(query, *args, **kwargs):
    thought = args[0].log
    action = args[0].tool
    action_input = args[0].tool_input
    agent_outputs.append((query, thought, action, action_input))

  


llm = OpenAI()
memory = ConversationBufferMemory(memory_key="chat_history")
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True, memory=memory
)

sql_query = None

def format_agent_outputs(agent_outputs,latest_query_output=False):
    agent_outputs_str = ""
    last_query = None
    if latest_query_output:
        last_query = agent_outputs[-1][0]
        for i, (query, thought, action, action_input) in enumerate(agent_outputs):
            if query != last_query:
                continue
            if thought == "" or thought is None:
                thought = "No thought"
            # check if atring contains word QUERY:
            if "Query: " not in agent_outputs_str:
                agent_outputs_str += f"Query: {query}\n\n\t"
                j = 0
            thought = re.split(r"(\\n)*Action(:)*", thought)[0]
            agent_outputs_str += f"Run {j}:\n\t\tThought: {thought}\n\t\tAction: {action}\n\t\tAction Input: {action_input}\n\n\t"
            j += 1
        return agent_outputs_str , action_input

    for i, (query, thought, action, action_input) in enumerate(agent_outputs):
        if query != last_query:
            if thought == "" or thought is None:
                thought = "No thought"
            agent_outputs_str += f"Query: {query}\n\n\t"
            last_query = query
            j = 0
        thought = re.split(r"(\\n)*Action(:)*", thought)[0]
        agent_outputs_str += f"Run {j}:\n\t\tThought: {thought}\n\t\tAction: {action}\n\t\tAction Input: {action_input}\n\n\t"
        j += 1
    return agent_outputs_str


def execute_agent(query):
    with get_openai_callback() as cb:
        cb.on_agent_action = lambda *args, **kwargs: new_agent_action(query, *args, **kwargs)
        output = agent_executor(inputs=query)
    return output

def answer(query, history):
    global sql_query
    history = history or []
    output = execute_agent(query)
    history.append((query, output))

    latest_query_output , sql_query = format_agent_outputs(agent_outputs, latest_query_output=True)
    data = execute_sql_query(sql_query)
    plot_data(data)
    plot_image_path = get_plot_image()
    update_plot_image(plot_image_path)
    return [(m[1]["input"], m[1]["output"]) for m in history], history,  latest_query_output , plot_image_path

from sqlalchemy import text

def execute_sql_query(sql_query):
    if sql_query is not None:
        try:
            # Wrap the query with the text() function
            executable_query = text(sql_query)
            result = connection.execute(executable_query)
            data = result.fetchall()
            print(data)
            return data
        except Exception as e:
            print(e)
            return []  # Return an empty list when an exception occurs
    return None

def plot_data(data):
    if data is not None:
        plt.figure()
        try:
            if len(data) > 0:  # If data is not empty, plot the data
                x = [row[0] for row in data]
                y = [row[1] for row in data]

                plt.plot(x, y)
                plt.xlabel('X-axis')
                plt.ylabel('Y-axis')
            else:  # If data is empty, display "No relevant plots available"
                plt.text(0.5, 0.5, 'No relevant plots available', fontsize=12, ha='center', va='center')
        except:
            plt.text(0.5, 0.5, 'No relevant plots available', fontsize=12, ha='center', va='center')
        plt.savefig('plot.png')


# Add Image component to Gradio interface
plot_image = gr.Image()

# Update Gradio interface with the plot image
def update_plot_image(image_path, delete_old_image=False):
    if delete_old_image:
        old_image_path = get_plot_image()
        if old_image_path is not None and os.path.exists(old_image_path):
            os.remove(old_image_path)
    plot_image.update(image_path)


def get_plot_image():
    return 'plot.png'

with gr.Blocks() as app:
    history_state = gr.State()

    with gr.Row():
        
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(label="Chatbot")

        with gr.Column(scale=5):
            current_query_heading = gr.Markdown("## Current Query")
            latest_query_output = gr.Textbox(label="Latest Query Output", lines=2, readonly=True)
            current_query_heading = gr.Markdown("## Relevant Plot")
            plot_image = gr.Image()  # Add Image component to Gradio interface
            
    with gr.Row():
        message = gr.Textbox(label="what's on your mind??",
                             placeholder="What is the answer to life, the universe and everything?",
                             lines=1)
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    message.submit(answer, inputs=[message, history_state],
                   outputs=[chatbot, history_state, latest_query_output, plot_image])
    submit.click(answer, inputs=[message, history_state],
                 outputs=[chatbot, history_state, latest_query_output, plot_image])

app.launch(debug=True)


