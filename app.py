from flask import Flask, render_template, request, url_for, session
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

app = Flask(__name__)
app.secret_key = "your_secret_key_here"

def generate_data(N, mu, beta0, beta1, sigma2, S):
    X = np.random.rand(N)
    Y = beta0 + beta1 * X + mu + np.random.normal(0, np.sqrt(sigma2), N)
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), Y)
    slope = model.coef_[0]
    intercept = model.intercept_

    plt.figure()
    plt.scatter(X, Y, color="black", label="Data Points")
    plt.plot(X, model.predict(X.reshape(-1, 1)), color="red", label=f"y = {slope:.2f}x + {intercept:.2f}")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Scatter Plot with Regression Line")
    plt.legend()
    plot1_path = "static/plot1.png"
    plt.savefig(plot1_path)
    plt.close()

    slopes = []
    intercepts = []
    for _ in range(S):
        X_sim = np.random.rand(N)
        Y_sim = beta0 + beta1 * X_sim + mu + np.random.normal(0, np.sqrt(sigma2), N)
        sim_model = LinearRegression()
        sim_model.fit(X_sim.reshape(-1, 1), Y_sim)
        slopes.append(sim_model.coef_[0])
        intercepts.append(sim_model.intercept_)

    plt.figure(figsize=(10, 5))
    plt.hist(slopes, bins=20, alpha=0.5, color="blue", label="Slopes")
    plt.hist(intercepts, bins=20, alpha=0.5, color="orange", label="Intercepts")
    plt.axvline(slope, color="blue", linestyle="--", linewidth=1, label=f"Slope: {slope:.2f}")
    plt.axvline(intercept, color="orange", linestyle="--", linewidth=1, label=f"Intercept: {intercept:.2f}")
    plt.title("Histogram of Slopes and Intercepts")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot2_path = "static/plot2.png"
    plt.savefig(plot2_path)
    plt.close()

    slope_more_extreme = sum(s >= slope for s in slopes) / S
    intercept_extreme = sum(i <= intercept for i in intercepts) / S

    return X, Y, slope, intercept, plot1_path, plot2_path, slope_more_extreme, intercept_extreme, slopes, intercepts

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        N = int(request.form["N"])
        mu = float(request.form["mu"])
        sigma2 = float(request.form["sigma2"])
        beta0 = float(request.form["beta0"])
        beta1 = float(request.form["beta1"])
        S = int(request.form["S"])

        X, Y, slope, intercept, plot1, plot2, slope_extreme, intercept_extreme, slopes, intercepts = generate_data(N, mu, beta0, beta1, sigma2, S)

        session["X"] = X.tolist()
        session["Y"] = Y.tolist()
        session["slope"] = slope
        session["intercept"] = intercept
        session["slopes"] = slopes
        session["intercepts"] = intercepts
        session["slope_extreme"] = slope_extreme
        session["intercept_extreme"] = intercept_extreme
        session["N"] = N
        session["mu"] = mu
        session["sigma2"] = sigma2
        session["beta0"] = beta0
        session["beta1"] = beta1
        session["S"] = S

        return render_template(
            "index.html",
            plot1=plot1,
            plot2=plot2,
            slope_extreme=slope_extreme,
            intercept_extreme=intercept_extreme,
            N=N,
            mu=mu,
            sigma2=sigma2,
            beta0=beta0,
            beta1=beta1,
            S=S,
        )
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    return index()

@app.route("/hypothesis_test", methods=["POST"])
def hypothesis_test():
    N = int(session.get("N"))
    S = int(session.get("S"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))

    parameter = request.form.get("parameter")
    test_type = request.form.get("test_type")

    if parameter == "slope":
        simulated_stats = np.array(slopes)
        observed_stat = slope
        hypothesized_value = beta1
    else:
        simulated_stats = np.array(intercepts)
        observed_stat = intercept
        hypothesized_value = beta0

    if test_type == ">":
        p_value = np.mean(simulated_stats >= observed_stat)
    elif test_type == "<":
        p_value = np.mean(simulated_stats <= observed_stat)
    elif test_type == "≠":
        extreme_diff = np.abs(observed_stat - hypothesized_value)
        p_value = np.mean(
            (simulated_stats >= hypothesized_value + extreme_diff)
            | (simulated_stats <= hypothesized_value - extreme_diff)
        )

    fun_message = "Wow! You've encountered a rare event!" if p_value <= 0.0001 else None

    plt.figure(figsize=(8, 5))
    plt.hist(simulated_stats, bins=30, color="gray", alpha=0.5, label=f"Simulated {parameter.capitalize()}")
    plt.axvline(observed_stat, color="blue", linestyle="--", linewidth=1, label=f"Observed {parameter.capitalize()}: {observed_stat:.2f}")
    plt.axvline(hypothesized_value, color="red", linestyle="--", linewidth=1, label=f"Hypothesized {parameter.capitalize()}: {hypothesized_value:.2f}")
    plt.title(f"Histogram of Simulated {parameter.capitalize()}s")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plot3_path = "static/plot3.png"
    plt.savefig(plot3_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot3=plot3_path,
        parameter=parameter,
        observed_stat=observed_stat,
        hypothesized_value=hypothesized_value,
        N=N,
        beta0=beta0,
        beta1=beta1,
        S=S,
        p_value=p_value,
        fun_message=fun_message,
    )

@app.route("/confidence_interval", methods=["POST"])
def confidence_interval():
    N = int(session.get("N"))
    mu = float(session.get("mu"))
    sigma2 = float(session.get("sigma2"))
    beta0 = float(session.get("beta0"))
    beta1 = float(session.get("beta1"))
    S = int(session.get("S"))
    X = np.array(session.get("X"))
    Y = np.array(session.get("Y"))
    slope = float(session.get("slope"))
    intercept = float(session.get("intercept"))
    slopes = session.get("slopes")
    intercepts = session.get("intercepts")

    parameter = request.form.get("parameter")
    confidence_level = float(request.form.get("confidence_level"))

    if parameter == "slope":
        estimates = np.array(slopes)
        observed_stat = slope
        true_param = beta1
    else:
        estimates = np.array(intercepts)
        observed_stat = intercept
        true_param = beta0

    mean_estimate = np.mean(estimates)
    std_estimate = np.std(estimates, ddof=1)
    t_critical = stats.t.ppf((1 + confidence_level / 100) / 2, df=S - 1)
    margin_of_error = t_critical * std_estimate / np.sqrt(S)
    ci_lower = mean_estimate - margin_of_error
    ci_upper = mean_estimate + margin_of_error
    includes_true = ci_lower <= true_param <= ci_upper

    plt.figure(figsize=(8, 5))
    plt.scatter(estimates, [0] * len(estimates), color="gray", alpha=0.5, label="Simulated Estimates")
    plt.scatter(mean_estimate, 0, color="blue", s=100, label="Mean Estimate")
    plt.hlines(y=0, xmin=ci_lower, xmax=ci_upper, color="blue", linewidth=3, label=f"{confidence_level}% Confidence Interval")
    plt.axvline(true_param, color="green", linestyle="--", linewidth=2, label="True Parameter")
    plt.title(f"{confidence_level}% Confidence Interval for {parameter.capitalize()} (Mean Estimate)")
    plt.xlabel(f"{parameter.capitalize()} Estimate")
    plt.yticks([])
    plt.legend()
    plot4_path = "static/plot4.png"
    plt.savefig(plot4_path)
    plt.close()

    return render_template(
        "index.html",
        plot1="static/plot1.png",
        plot2="static/plot2.png",
        plot4=plot4_path,
        parameter=parameter,
        confidence_level=confidence_level,
        mean_estimate=mean_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        includes_true=includes_true,
        observed_stat=observed_stat,
        N=N,
        mu=mu,
        sigma2=sigma2,
        beta0=beta0,
        beta1=beta1,
        S=S,
    )

if __name__ == "__main__":
    app.run(debug=True, port=5050)
