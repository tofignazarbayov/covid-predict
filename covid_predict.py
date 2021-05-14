import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import mean_squared_error, r2_score
import urllib.request as request
import datetime
from numpy.polynomial import Polynomial as poly

#np.seterr(all="ignore") #to ignore overflow error in np.exp

plt.style.use("dark_background") #to make graph's background black

def prediction(country="world", how_far=1, fittingtype="polynomial"):
    """
    Function takes 3 arguments:

    country -> takes string, "world" to do prediction worldwide, and the name of specific countries (matching with names in dataset of Johns Hopkins university) to predict for specific country
    how_far -> takes integer, number of how many days into the future you would like to do predictions
    fittingtype -> takes string of the type of fitting, there are 3: 
                    "polynomial" (good to do predictions for tomorrow, as high order polynomial reflects small variations in the data), based on polynomial fitting, 
                    "exponential" (I don't actually like it, fits bad), based on exponential fitting, 
                    "s_shape" (good to predict general trend, many days into the future), based on S shape fitting or sigmoid fitting (but unbouded sigmoid, which is not locked in 0 to 1 cage), 
                    "best", based on running all 3 models and choosing the best by finding the model with smallest root mean squared error, 
                    "best_of_2", based on mixing polynomial fitting model with S shape fitting model by using inverse of their mean root mean square errors as their coefficients in finding value
    """

    #Importing data
    if country.lower() == "world":
        data = pd.read_csv(request.urlopen('https://raw.githubusercontent.com/datasets/covid-19/master/data/worldwide-aggregated.csv'))
        #print(data.dtypes)

    else:
        data = pd.read_csv(request.urlopen('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'))
        #print(data.dtypes)
        countries_list = sorted(list(set(data["Country"])))
        #print(countries_list)
    
        #Choosing country for analysis
        data = data.loc[data["Country"] == country]
    
    #Preparing data for curve fitting
    days_country=np.array(np.arange(1, len(data)+1), dtype=int)
    cases_country=np.array(data["Confirmed"], dtype=int)
    #print(f"Shape of input {days_country.shape}")
    #print(f"Shape of output {cases_country.shape}")
    
    if how_far == 1:
        how_far_str = "by tomorrow"
        how_far+=1 #because data comes one day late (data for today will come tomorrow)
    elif how_far == 7:
        how_far_str = "after one week"
        how_far+=1 #because data comes one day late (data for today will come tomorrow)
    else:
        how_far_str = f"{how_far} days from today"
        how_far+=1 #because data comes one day late (data for today will come tomorrow)

    if fittingtype == "polynomial":
        #Polynomial curve fitting
        tmp = 0
        power = [0] #because assigning a value to variable inside if conditional is a problem. it will continute to use global copy with initial value. assigning a value to element in a list uses address, so making assignment inside if conditional work properly
        for i in range(1, 35):
            #predictor = np.poly1d(np.polyfit(days_country,cases_country,i)) #old way to use polynomials. showed error in high orders
            predictor = poly.fit(days_country, cases_country, i) #fitting polynomial of power i
            today = int(cases_country[len(cases_country)-1]) #number of cases today in the specific country
            tomorrow = int(predictor(np.array([len(days_country)+how_far]))) #number of cases predicted how_far number of days into the future
            yesterday = int(cases_country[len(cases_country)-1-how_far]) #number of cases how_far number of days in the past
            if (tomorrow >= today) and ((tomorrow - today) < int(1.2*(today - yesterday))): #ensuring that predicted number is higher or equal than today's number and difference between value predicted for how_far days into the future and today is smaller than 120% of the difference between today's value and value how_far days in the past
                tmp = int(predictor(np.array([len(days_country)+how_far]))) #saving model as temporarly the best
                power[0] = i #saving its power for its future initialization in case if its really best
        
        predictor = poly.fit(days_country, cases_country, power[0])

        time_list = sorted(list(set(data["Date"]))) #used for creating a time message to show when data was last updated
        data_time = str(time_list[len(time_list)-1])

        message = f"There will be {int(predictor(np.array([len(days_country)+how_far])))} people ill {how_far_str} in {country}\nPrediction was estimated by polynom of power {power[0]}\nData updated at {data_time}"
        print(message)
    
    elif fittingtype == "exponential":
        #Exponential curve fitting
        def func(x, a, b, c):
            return a+np.exp(b*x+c)
        params, pcov = scipy.optimize.curve_fit(func, days_country, cases_country, maxfev=10000000)
        #print(params)

        def predictor(x, a=params[0], b=params[1], c=params[2]):
            return a+np.exp(b*x+c)

        time_list = sorted(list(set(data["Date"]))) #used for creating a time message to show when data was last updated
        data_time = str(time_list[len(time_list)-1])

        message = f"There will be {int(predictor(np.array([len(days_country)+how_far])))} people ill {how_far_str} in {country}\nPrediction was estimated by exponential function {round(params[0], 3)} + exp({round(params[1], 3)} * x + {round(params[2], 3)})\nData updated at {data_time}"
        print(message)
    
    elif fittingtype == "s_shape":
        #S shape curve fitting
        def func(x, a, b, c, d):
            return a/(b+np.exp(-1.*c*(x-d)))
        params, pcov = scipy.optimize.curve_fit(func, days_country, cases_country, maxfev=10000000)
        #print(params)

        def predictor(x, a=params[0], b=params[1], c=params[2], d=params[3]):
            return a/(b+np.exp(-1.*c*(x-d)))
        
        time_list = sorted(list(set(data["Date"]))) #used for creating a time message to show when data was last updated
        data_time = str(time_list[len(time_list)-1])

        message = f"There will be {int(predictor(np.array([len(days_country)+how_far])))} people ill {how_far_str} in {country}\nPrediction was estimated by S shape function {round(params[0], 3)} / ({round(params[1], 3)} + exp(-{round(params[2], 3)} * (x - {round(params[3], 3)})))\nData updated at {data_time}"
        print(message)

    elif fittingtype == "best":
        #Polynomial curve fitting
        tmp = 0
        power = [0]
        for i in range(1, 35):
            #predictor_poly = np.poly1d(np.polyfit(days_country,cases_country,i))
            predictor_poly = poly.fit(days_country, cases_country, i)
            today = int(cases_country[len(cases_country)-1])
            tomorrow = int(predictor_poly(np.array([len(days_country)+how_far])))
            yesterday = int(cases_country[len(cases_country)-1-how_far])
            if (tomorrow >= today) and ((tomorrow - today) < int(1.2*(today - yesterday))):
                tmp = int(predictor_poly(np.array([len(days_country)+how_far])))
                power[0] = i
        
        predictor_poly = poly.fit(days_country, cases_country, power[0])

        rmse_poly = np.sqrt(mean_squared_error(cases_country, predictor_poly(days_country)))

        #Exponential curve fitting
        def func(x, a, b, c):
            return a+np.exp(b*x+c)
        params, pcov = scipy.optimize.curve_fit(func, days_country, cases_country, maxfev=10000000)
        #print(params)

        def predictor_exp(x, a=params[0], b=params[1], c=params[2]):
            return a+np.exp(b*x+c)
        
        rmse_exp = np.sqrt(mean_squared_error(cases_country, predictor_exp(days_country)))

        #S shape curve fitting
        def func(x, a, b, c, d):
            return a/(b+np.exp(-1.*c*(x-d)))
        params, pcov = scipy.optimize.curve_fit(func, days_country, cases_country, maxfev=10000000)
        #print(params)

        def predictor_s(x, a=params[0], b=params[1], c=params[2], d=params[3]):
            return a/(b+np.exp(-1.*c*(x-d)))

        rmse_s = np.sqrt(mean_squared_error(cases_country, predictor_s(days_country)))

        if rmse_poly<rmse_exp and rmse_poly<rmse_s:
            predictor = predictor_poly

            time_list = sorted(list(set(data["Date"]))) #used for creating a time message to show when data was last updated
            data_time = str(time_list[len(time_list)-1])
            
            message = f"There will be {int(predictor_poly(np.array([len(days_country)+how_far])))} people ill {how_far_str} in {country}\nPrediction was estimated by polynom of power {power[0]}\nData updated at {data_time}"
            print(message)

        elif rmse_s<rmse_exp and rmse_s<rmse_poly:
            predictor = predictor_s

            time_list = sorted(list(set(data["Date"]))) #used for creating a time message to show when data was last updated
            data_time = str(time_list[len(time_list)-1])

            message = f"There will be {int(predictor_s(np.array([len(days_country)+how_far])))} people ill {how_far_str} in {country}\nPrediction was estimated by S shape function {round(params[0], 3)} / ({round(params[1], 3)} + exp(-{round(params[2], 3)} * (x - {round(params[3], 3)})))\nData updated at {data_time}"
            print(message)
        
        else:
            predictor = predictor_exp

            time_list = sorted(list(set(data["Date"]))) #used for creating a time message to show when data was last updated
            data_time = str(time_list[len(time_list)-1])

            message = f"There will be {int(predictor_exp(np.array([len(days_country)+how_far])))} people ill {how_far_str} in {country}\nPrediction was estimated by exponential function {round(params[0], 3)} + exp({round(params[1], 3)} * x + {round(params[2], 3)})\nData updated at {data_time}"
            print(message)
    
    elif fittingtype == "best_of_2":
        #Polynomial curve fitting
        tmp = 0
        power = [0]
        for i in range(1, 35):
            #predictor_poly = np.poly1d(np.polyfit(days_country,cases_country,i))
            predictor_poly = poly.fit(days_country, cases_country, i)
            today = int(cases_country[len(cases_country)-1])
            tomorrow = int(predictor_poly(np.array([len(days_country)+how_far])))
            yesterday = int(cases_country[len(cases_country)-1-how_far])
            if (tomorrow >= today) and ((tomorrow - today) < int(1.2*(today - yesterday))):
                tmp = int(predictor_poly(np.array([len(days_country)+how_far])))
                power[0] = i
        
        predictor_poly = poly.fit(days_country, cases_country, power[0])
        rmse_poly = np.sqrt(mean_squared_error(cases_country, predictor_poly(days_country)))
        r2_poly = r2_score(cases_country, predictor_poly(days_country))
        model_poly = f"polynom of power {power[0]}"

        #S shape curve fitting
        def func(x, a, b, c, d):
            return a/(b+np.exp(-1.*c*(x-d)))
        params, pcov = scipy.optimize.curve_fit(func, days_country, cases_country, maxfev=10000000)
        #print(params)

        def predictor_s(x, a=params[0], b=params[1], c=params[2], d=params[3]):
            return a/(b+np.exp(-1.*c*(x-d)))
        
        rmse_s = np.sqrt(mean_squared_error(cases_country, predictor_s(days_country)))
        r2_s = r2_score(cases_country, predictor_s(days_country))
        model_s = f"S shape function {round(params[0], 3)} / ({round(params[1], 3)} + exp(-{round(params[2], 3)} * (x - {round(params[3], 3)})))"

        #coef_poly = 1-(rmse_poly/(rmse_exp + rmse_poly)) #the other way to do the calculations
        #coef_exp = 1-(rmse_exp/(rmse_exp + rmse_poly))

        e_poly = rmse_poly/((rmse_s + rmse_poly)/2)
        e_s = rmse_s/((rmse_s + rmse_poly)/2)

        #coef_poly = e_exp
        #coef_exp = e_poly

        coef_poly = 1/e_poly
        coef_s = 1/e_s

        #print(coef_poly)
        #print(coef_s)
        
        #Chossing predictor for future graph
        #predictor = predictor_s
        if rmse_poly<rmse_s:
            predictor = predictor_poly

        else:
            predictor = predictor_s

        value_poly = int(predictor_poly(np.array([len(days_country)+how_far])))
        value_s = int(predictor_s(np.array([len(days_country)+how_far])))
        value = int((coef_poly*value_poly+coef_s*value_s)/(coef_poly+coef_s))

        time_list = sorted(list(set(data["Date"]))) #used for creating a time message to show when data was last updated
        data_time = str(time_list[len(time_list)-1])
            
        message = f"There will be {value} people ill {how_far_str} in {country}\nPrediction was estimated by combined model of {model_poly} and {model_s}\nData updated at {data_time}"
        #message = f"There will be {value} people ill {how_far_str} in {country}\nPrediction was estimated by combined model\nData updated at {data_time}"
        print(message)

    #Creating graph
    plt.figure(figsize=(10,5))
    plt.xlabel("Days")
    plt.ylabel("Cases")
    plt.title(f"Prediction graph for {country}")

    plt.scatter(days_country, cases_country, color="purple", label = "Real data")
    plt.plot(days_country, predictor(days_country), color="orange", label = "Prediction")
    plt.legend(prop={'size': 15})
    time = datetime.datetime.now().strftime("%d%b%Y%H%M")
    graph = f"prediction_{country}_{how_far}_{fittingtype}_{time}.png"
    plt.savefig(graph)
    #plt.show()
    
    #Evaluating errors
    rmse = np.sqrt(mean_squared_error(cases_country, predictor(days_country)))
    r2 = r2_score(cases_country, predictor(days_country))
    #print(f"RMSE error is {rmse}")
    #print(f"R2 error is {r2}")

    result = {"message":message, "graph":graph, "rmse":rmse, "r2":r2}
    return result

def situation(country="world"):
    """
    Function takes 3 arguments:

    country -> takes string, "world" to do prediction worldwide, and the name of specific countries (matching with names in dataset of Johns Hopkins university) to describe specific country
    """
    #Importing data
    if country.lower() == "world":
        data = pd.read_csv(request.urlopen('https://raw.githubusercontent.com/datasets/covid-19/master/data/worldwide-aggregated.csv'))
        #print(data.dtypes)

    else:
        data = pd.read_csv(request.urlopen('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'))
        #print(data.dtypes)
        countries_list = sorted(list(set(data["Country"])))
        #print(countries_list)
    
        #Choosing country for analysis
        data = data.loc[data["Country"] == country]
    
    days_country=np.array(np.arange(1, len(data)+1), dtype=int)
    cases_country=np.array(data["Confirmed"], dtype=int)
    deaths_country=np.array(data["Deaths"], dtype=int)
    recovered_country=np.array(data["Recovered"], dtype=int)
    
    today_cases = cases_country[len(cases_country)-1]
    today_deaths = deaths_country[len(deaths_country)-1]
    today_recovered = recovered_country[len(recovered_country)-1]

    time_list = sorted(list(set(data["Date"]))) #used for creating a time message to show when data was last updated
    data_time = str(time_list[len(time_list)-1])

    message = f"Situation in {country} today:\n{today_cases} cases\n{today_deaths} deaths\n{today_recovered} recovered\nData updated at {data_time}"
    print(message)

    #Graph of trend
    plt.figure(figsize=(10,5))
    plt.xlabel("Days")
    plt.ylabel("Cases")
    plt.title(f"Trend graph for {country}")
    plt.plot(days_country, cases_country, color="red", label = "Cases")
    plt.plot(days_country, deaths_country, color="white", label = "Deaths")
    plt.plot(days_country, recovered_country, color="green", label = "Recovered")
    plt.legend(prop={'size': 15})
    time = datetime.datetime.now().strftime("%d%b%Y%H%M")
    graph = f"situation_{country}_{time}.png"
    plt.savefig(graph)
    #plt.show()

    result = {"message":message, "graph":graph}
    return result

def top10():
    #Importing data
    data = pd.read_csv(request.urlopen('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv'))
    #print(data.dtypes)
    countries_list = sorted(list(set(data["Country"])))
    #print(countries_list)

    time_list = sorted(list(set(data["Date"]))) #list of all the dates without repetition
    data_today = data.loc[data["Date"] == time_list[len(time_list)-1]] #filterting array by today's date
    data_today.sort_values(by=["Confirmed"], inplace=True, ascending = False) #sorting values in decreasing number of cases
    countries_top = np.array(data_today["Country"])[:10] #taking top 10 country names with the highest number of cases
    cases_top = np.array(data_today["Confirmed"])[:10] #taking top 10 number of cases
    deaths_top = np.array(data_today["Deaths"])[:10] #taking top 10 number of deaths
    recovered_top = np.array(data_today["Recovered"])[:10] #taking top 10 number of recovered
    #print(countries_top)

    #Creating bar graph
    countries_position = np.arange(10, 0, -1)
    plt.figure(figsize=(10,5))
    plt.barh(countries_position, cases_top, color="orangered", label = "Cases")
    plt.barh(countries_position, deaths_top, color="white", label = "Deaths")
    plt.barh(countries_position, recovered_top, color="dodgerblue", label = "Recovered")
    plt.yticks(countries_position, countries_top)
    time = datetime.datetime.now().strftime("%d%b%Y%H%M")
    plt.legend(prop={'size': 15})
    graph = f"top10_{time}.png"
    plt.savefig(graph)
    result = {"graph":graph}
    return result

