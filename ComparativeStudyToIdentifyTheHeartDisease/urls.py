"""medical3 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from ComparativeStudyToIdentifyTheHeartDisease import views as mainView
from admins import views as admins
from users import views as usr

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", mainView.index, name="index"),
    path("AdminLogin/", mainView.AdminLogin, name="AdminLogin"),
    path("UserLogin/", mainView.UserLogin, name="UserLogin"),
    path("UserRegister/", mainView.UserRegister, name="UserRegister"),

    # adminviews
    path("AdminLoginCheck/", admins.AdminLoginCheck, name="AdminLoginCheck"),
    path("AdminHome/", admins.AdminHome, name="AdminHome"),
    path('RegisterUsersView/', admins.RegisterUsersView, name='RegisterUsersView'),
    path('ActivaUsers/', admins.ActivaUsers, name='ActivaUsers'),

    # User Views

    path("UserRegisterActions/", usr.UserRegisterActions, name="UserRegisterActions"),
    path("UserLoginCheck/", usr.UserLoginCheck, name="UserLoginCheck"),
    path("UserHome/", usr.UserHome, name="UserHome"),
    path("View_DATA/", usr.View_DATA, name="View_DATA"),
    path("ML/", usr.ML, name="ML"),
    path('MLResult/', usr.MLResult, name="MLResult"),
    path('RandomForest/', usr.RandomForest, name="RandomForest"),
    path('SVM/', usr.SVM, name="SVM"),
    path('LR/', usr.LR, name="LR"),
    path('NB/', usr.NB, name="NB"),
    path('DTC/', usr.DTC, name="DTC"),
    path('KNN/', usr.KNN, name="KNN"),
    path("Graph/", usr.Graph, name="Graph")

]
