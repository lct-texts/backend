from rest_framework.generics import ListCreateAPIView, RetrieveAPIView, ListAPIView, UpdateAPIView
from .serializers import ConversationSerializer, CoordsSerializer, ConversationUpdateSerializer
from ..models import Conversation, Coords
from rest_framework.pagination import PageNumberPagination


class ListConversationView(ListCreateAPIView):
    model = Conversation
    queryset = Conversation.objects.all().reverse()
    serializer_class = ConversationSerializer


class RetrieveConversationView(RetrieveAPIView):
    model = Conversation
    queryset = Conversation.objects.all()
    serializer_class = ConversationSerializer


class LargeResultsSetPagination(PageNumberPagination):
    page_size = 1000
    page_size_query_param = 'page_size'
    max_page_size = 10000


class ListSentimentCordsView(ListAPIView):
    model = Coords
    serializer_class = CoordsSerializer
    queryset = Coords.objects.filter(type='sentiment')
    pagination_class = None


class ListStopThemeCordsView(ListAPIView):
    model = Coords
    serializer_class = CoordsSerializer
    queryset = Coords.objects.filter(type='stop_theme')
    pagination_class = None


class ConversationUpdateAPIView(UpdateAPIView):
    model = Conversation
    serializer_class = ConversationUpdateSerializer
    queryset = Conversation.objects.all()
