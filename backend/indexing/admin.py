from django.contrib import admin
from .models import Conversation, Message

admin.site.register([Conversation, Message])

"""
from backend.indexing.models import Message, Conversation
import pickle
with open('../message.pickle', 'rb') as file:
    msg_data = pickle.load(file)
with open('../conversations.pickle', 'rb') as file:
    conv_data = pickle.load(file)


for item in conv_data:
    conv = Conversation()
    conv.from_dict(item)

for item in msg_data:
    conv = Message()
    conv.from_dict(item)


"""