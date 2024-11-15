from django.shortcuts import render
# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.shortcuts import HttpResponse
from .algorithms.ProcessAlgorithm import Algorithms

algo = Algorithms()


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)

                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def View_DATA(request):
    import pandas as pd
    from django.conf import settings
    import os
    path = os.path.join(settings.MEDIA_ROOT, 'heart1.csv')
    df = pd.read_csv(path)
    df = df[
        ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal',
         'target']]
    df = df.head(310).to_html
    return render(request, 'users/readDataset.html', {'data': df})


def ML(request):
    return render(request, 'users/ML.html')


def MLResult(request):
    if request.method == "POST":
        from django.conf import settings
        import pandas as pd
        age = request.POST.get('age')

        gender = request.POST.get('sex')
        cp = request.POST.get('cp')
        trestbps = request.POST.get('trestbps')
        chol = request.POST.get('chol')
        fbs = request.POST.get('fbs')
        restecg = request.POST.get('restecg')
        thalach = request.POST.get('thalach')
        exang = request.POST.get('exang')
        oldpeak = request.POST.get('oldpeak')
        slope = request.POST.get('slope')
        ca = request.POST.get('ca')
        thal = request.POST.get('thal')

        path = settings.MEDIA_ROOT + "\\" + "heart1.csv"
        data = pd.read_csv(path, delimiter=',')
        x = data.iloc[:, 0:13]
        print(x)
        y = data.iloc[:, 13]
        print(y)
        x = pd.get_dummies(x)
        print(x)

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)
        print(x_test)
        x_train = pd.DataFrame(x_train)
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        print('x-train:', x_train)
        test_set = [age, gender, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        print('y_train:', y_train)
        model.fit(x_train, y_train)
        y_pred = model.predict([test_set])
        print(y_pred)

        if y_pred == 0:
            msg = "you'r health is good"
        elif y_pred == 1:
            msg = "you'r health is bad"
        return render(request, 'users/ML.html', {'msg': msg})


def RandomForest(request):
    dt_acc, dt_recall, dt_precc, dt_f1 = algo.RandomForest()
    print("recall:", dt_recall)
    return render(request, 'users/RandomForest.html',
                  {'dt_acc': dt_acc, 'dt_recall': dt_recall, 'dt_precc': dt_precc, 'dt_f1': dt_f1})


def SVM(request):
    dt_acc, dt_recall, dt_precc, dt_f1 = algo.SVM()
    print("recall:", dt_recall)
    return render(request, 'users/SVM.html',
                  {'dt_acc': dt_acc, 'dt_recall': dt_recall, 'dt_precc': dt_precc, 'dt_f1': dt_f1})


def LR(request):
    dt_acc, dt_recall, dt_precc, dt_f1 = algo.LogisticRegression()
    print("recall:", dt_recall)
    return render(request, 'users/LR.html',
                  {'dt_acc': dt_acc, 'dt_recall': dt_recall, 'dt_precc': dt_precc, 'dt_f1': dt_f1})


def NB(request):
    dt_acc, dt_recall, dt_precc, dt_f1 = algo.NaiveBayes()
    print("recall:", dt_recall)
    return render(request, 'users/NB.html',
                  {'dt_acc': dt_acc, 'dt_recall': dt_recall, 'dt_precc': dt_precc, 'dt_f1': dt_f1})


def DTC(request):
    dt_acc, dt_recall, dt_precc, dt_f1 = algo.DesicionTree()
    print("recall:", dt_recall)
    return render(request, 'users/DT.html',
                  {'dt_acc': dt_acc, 'dt_recall': dt_recall, 'dt_precc': dt_precc, 'dt_f1': dt_f1})


def KNN(request):
    dt_acc, dt_recall, dt_precc, dt_f1 = algo.KNeighbors()
    print("recall:", dt_recall)
    return render(request, 'users/KN.html',
                  {'dt_acc': dt_acc, 'dt_recall': dt_recall, 'dt_precc': dt_precc, 'dt_f1': dt_f1})


def Graph(request):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    df = pd.DataFrame({"Accuracy": [85, 85, 80, 86, 82, 78], "precision": [88, 83, 81, 86, 81, 81],
                       "recall": [86, 93, 83, 90, 81, 86], "f1": [87, 87, 82, 88, 85, 81]},
                      index=["RF", "SVM", "DTC", "KNN", "LR", "NB"])
    print(df)
    ax = df.plot(kind="bar", figsize=(10, 5))
    plt.title("Accuracy,Precision,Recall,F1 Scores")
    plt.xlabel("ALGORITHMS")
    plt.ylabel("Acc,Prec,rec,F1")
    plt.show()

    return render(request, "users/Graph.html", {"x": ax})
