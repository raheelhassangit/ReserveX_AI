from django.urls import path
from . import views

urlpatterns = [ 
    path("", views.landing, name="landing"),
    path("terms/", views.terms, name="terms"),
    path("privacy/", views.privacy, name="privacy"),
]