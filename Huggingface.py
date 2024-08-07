# coding: utf-8

# In[1]:


from transformers import pipeline


# # Sentiment analysis
# 
#    #### here we check if given sentence is negative or positive.

# In[2]:


classifier = pipeline(task="sentiment-analysis",model="distilbert-base-uncased-finetuned-sst-2-english")


# In[3]:


classifier('Love it.')


# In[6]:


text = ['I dont like it',       'I love it',       'Thanks for nothing']
classifier(text)


# In[5]:


classifier1 = pipeline(task="text-classification",model="cardiffnlp/twitter-roberta-base-sentiment-latest")


# In[10]:


classifier1(text)


# # Summarization

# In[20]:


summarizaiton = pipeline('summarization',model='facebook/bart-large-cnn')


# In[25]:


text1 = """In 2023, four criminal indictments were filed against Donald Trump, president of the United States from 2017 to 2021.
Two indictments are on state charges (one in New York and one in Georgia) and two indictments (as well as one superseding
indictment) are on federal charges (one in Florida and one in the District of Columbia).[1] The New York trial began on 
April 15,2024 and concluded on May 30, 2024 with Trumps conviction on all 34 charges. Sentencing is scheduled for September 18."""



# In[27]:


summarized_text = summarizaiton(text1, min_length = 5, max_length = 108)[0]['summary_text']
summarized_text


# # Conversational Pipeline

# In[29]:


chatbot = pipeline(model = "facebook/blenderbot-400M-distill")


# In[36]:


chatbot("Hey! whats going on")


# # Chatbot with Gradio

# In[ ]:


import gradio as gr


# In[55]:


message_list = []
response_list = []

def vanilla_bot(message, history):
    conversation = Conversation(text = message, past_user_inputs= message_list, generated_response = response_list)
    conversation = chatbot(conversation)
    return conversation.generated_response[-1]

demo = gr.ChatInterface(vanilla_bot, title = 'Vanilla chatbot', description = 'Enter the text to start conversation')
demo.launch()

