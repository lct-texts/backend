from django.db import models
from typing import Dict


class DictWritable:
    def from_dict(self, fields: Dict):
        for key, value in fields.items():
            setattr(self, key, value)
        self.save()


class Conversation(DictWritable, models.Model):
    SENTIMENT_CHOICES = (
        ('positive', 'positive'),
        ('neutral', 'neutral'),
        ('negative', 'negative')
    )
    sentiment = models.CharField(choices=SENTIMENT_CHOICES, max_length=100)
    sentiment_score = models.DecimalField(default=0, max_digits=3, decimal_places=2)
    can_promote = models.BooleanField(default=False)
    stop_theme = models.CharField(max_length=100, null=True)



class Message(DictWritable, models.Model):
    AUTHOR_CHOICES = (
        ('user', 'user'),
        ('bot', 'bot')
    )

    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE)
    author = models.CharField(choices=AUTHOR_CHOICES, max_length=100)
    message = models.CharField(max_length=1000)
    lemmatization_message = models.CharField(max_length=1000, null=True)
    stop_theme = models.CharField(null=True, max_length=100)
    sequence_number = models.IntegerField(null=True)


class Coords(models.Model):
    COORDS_TYPE = (
        ('stop_theme', 'stop_theme'),
        ('sentiment', 'sentiment')
    )

    message = models.ForeignKey(Message, on_delete=models.CASCADE)
    type = models.CharField(choices=COORDS_TYPE, max_length=100)
    x = models.DecimalField(max_digits=10, decimal_places=2)
    y = models.DecimalField(max_digits=10, decimal_places=2)
    label = models.CharField(max_length=100)

    @property
    def conversation_id(self):
        return self.message.conversation.id
    
    @property
    def message_text(self):
        return self.message.message
