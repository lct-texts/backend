from django.conf import settings
from rest_framework.routers import DefaultRouter, SimpleRouter
from django.urls import path
from backend.indexing.api.views import ListConversationView, RetrieveConversationView, ListSentimentCordsView, ListStopThemeCordsView, ConversationUpdateAPIView
from backend.users.api.views import UserViewSet

if settings.DEBUG:
    router = DefaultRouter()
else:
    router = SimpleRouter()

router.register("users", UserViewSet)


app_name = "api"
urlpatterns = router.urls

urlpatterns.extend([
    path('conversations', ListConversationView.as_view()),
    path('conversations/<pk>', RetrieveConversationView.as_view()),
    path('sentimental-cords', ListSentimentCordsView.as_view()),
    path('stopwords-cords', ListStopThemeCordsView.as_view()),
    path('conversations/update-state/<pk>', ConversationUpdateAPIView.as_view())
])
