from ..models import Conversation, Message, Coords
from rest_framework import serializers
from ..service import search, calculate_metrics_based_on_search, run_stop_theme_nearest
from collections import Counter


class MessageSerializer(serializers.ModelSerializer):
    class Meta:
        model = Message
        fields = ('author', 'message', 'sequence_number')
        read_only_fields = ('sequence_number',)


class ConversationSerializer(serializers.ModelSerializer):
    message_set = MessageSerializer(many=True)
    
    class Meta:
        model = Conversation
        fields = ('sentiment', 'message_set', 'sentiment_score', 'can_promote', 'id', 'stop_theme')
        read_only_fields = ('sentiment_score', 'can_promote', 'sentiment', 'id', 'stop_theme')
    

    def create(self, validated_data):
        conversation = Conversation.objects.create(sentiment='neutral')
        scores = []
        sentiments = []
        msgs = []
        for i, msg in enumerate(validated_data['message_set']):
            Message.objects.create(
                conversation=conversation,
                author=msg['author'],
                message=msg['message'],
                sequence_number=i
            )
            if msg['author'] == 'user':
                metrics = calculate_metrics_based_on_search(
                    search(msg['message'])
                )
                msgs.append(msg['message'])
                scores.append(metrics['score'])
                sentiments.append(metrics['sentiment'])
        conversation.sentiment_score = sum(scores) / len(scores)
        if sum(scores) / len(scores) < 0:
            conversation.sentiment = 'negative'
        elif sum(scores) / len(scores) == 0:
            conversation.sentiment = 'neutral'
        elif sum(scores) / len(scores) > 0:
            conversation.sentiment = 'positive'
        conversation.can_promote = (sum(scores) / len(scores)) > 0
        if sum(scores) / len(scores) < 0:
            conversation.stop_theme = run_stop_theme_nearest(msgs)
        conversation.save()
        return conversation


class ConversationUpdateSerializer(serializers.ModelSerializer):
    class Meta:
        model = Conversation
        fields = ('sentiment', 'stop_theme')


class CoordsSerializer(serializers.ModelSerializer):
    class Meta:
        model = Coords
        fields = ['message_text', 'type', 'x', 'y', 'label', 'conversation_id']