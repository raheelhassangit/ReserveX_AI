from django.shortcuts import render

def landing(request):
    return render(request, "landing.html")

def terms(request):
    return render(request, "terms.html")

def privacy(request):
    return render(request, "privacy.html")