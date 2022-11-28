import secrets
import asyncio
import os.path
import numpy as np
import pandas as pd

from shutil import copyfile
from flask import *
from causal.preprocessing import generic_preprocessing as gp
from causal.modules import logistic as lg
from causal.modules import naive_bayes as nb
from causal.modules import linear_svc as lsvc
from causal.modules import knn
from causal.modules import abc as causalg
from causal.modules import model_graph as gph
from causal.modules import decision_tree as dtree
from causal.modules import random_forest as rfc
from causal.visualization import visualize as vis
from causal.nocache import nocache
from causal import app

global posted
save_path = "causal/uploads/"
exts = ["csv", "json", "yaml"]
posted = 0


@app.route("/")
@app.route("/preprocess", methods=["GET", "POST"])
@nocache
def preprocess():

    if request.method == "POST":

        if request.form["Submit"] == "Upload":
            data = request.files["data"]
            ext = data.filename.split(".")[1]
            if ext in exts:
                session["ext"] = ext
                session["fname"] = data.filename
                data.save("causal/uploads/" + data.filename)
                df = gp.read_dataset("causal/uploads/" + data.filename)
                df.to_csv("causal/clean/clean.csv")
                session["haha"] = True
                flash(f"File uploaded successully", "success")
            else:
                flash(f"Upload Unsuccessful. Please try again", "danger")

        elif request.form["Submit"] == "Delete":
            try:
                df = gp.read_dataset("causal/clean/clean.csv")
                df = gp.delete_column(df, request.form.getlist("check_cols"))
                df.to_csv("causal/clean/clean.csv", mode="w", index=False)
                flash(f"Column(s) deleted Successfully", "success")
            except:
                flash(f"Error! Upload Dataset", "danger")

        elif request.form["Submit"] == "Clean":
            try:
                df = gp.read_dataset("causal/clean/clean.csv")
                # print(request.form["how"])
                if request.form["how"] is not "any":
                    df = gp.treat_missing_numeric(
                        df, request.form.getlist("check_cols"), how=request.form["how"]
                    )
                elif request.form["howNos"] is not None:
                    df = gp.treat_missing_numeric(
                        df,
                        request.form.getlist("check_cols"),
                        how=float(request.form["howNos"]),
                    )

                df.to_csv("causal/clean/clean.csv", mode="w", index=False)
                flash(f"Column(s) cleant Successfully", "success")
            except:
                flash(f"Error! Upload Dataset", "danger")

        elif request.form["Submit"] == "Visualize":
            global posted
            df = gp.read_dataset("causal/clean/clean.csv")

            x_col = request.form["x_col"]

            if vis.hist_plot(df, x_col):
                posted = 1

    if session.get("haha") is not None:
        df = gp.read_dataset("causal/clean/clean.csv")
        description = gp.get_description(df)
        columns = gp.get_columns(df)
        # print(columns)
        dim1, dim2 = gp.get_dim(df)
        head = gp.get_head(df)

        return render_template(
            "preprocess.html",
            active="preprocess",
            title="Preprocess",
            filename=session["fname"],
            posted=posted,
            no_of_rows=len(df),
            no_of_cols=len(columns),
            dim=str(dim1) + " x " + str(dim2),
            description=description.to_html(
                classes=[
                    "table-bordered",
                    "table-striped",
                    "table-hover",
                    "thead-light",
                ]
            ),
            columns=columns,
            head=head.to_html(
                classes=[
                    "table",
                    "table-bordered",
                    "table-striped",
                    "table-hover",
                    "thead-light",
                ]
            ),
        )
    else:
        return render_template(
            "preprocess.html", active="preprocess", title="Preprocess",
        )


@app.route("/graph", methods=["GET", "POST"])
def graph():
    if request.method == "POST":
        # target = request.form["target"]
        # gp.arrange_columns(target)
        graph = int(request.form["graph"])
        # hidden_val = int(request.form["hidden"])
        # scale_val = int(request.form["scale_hidden"])
        # encode_val = int(request.form["encode_hidden"])
        # columns = vis.get_columns()

        # if hidden_val == 0:
        #     data = request.files["choiceVal"]
        #     ext = data.filename.split(".")[1]
        #     if ext in exts:
        #         data.save("uploads/test." + ext)
        #     else:
        #         return "File type not accepted!"
        #     choiceVal = 0
        # else:
        #     choiceVal = int(request.form["choiceVal"])

        if graph == 0:
                gph.discover_bpmn_model()
                return render_template(
                    "classifier_page.html",
                    src="img/bpmn.png",
                )    
        elif graph == 1:
                gph.discover_pn_model()
                return render_template(
                    "classifier_page.html",
                    src="img/pnet.png",
                )

    #     elif classifier == 2:
    #         ret_vals = lsvc.lin_svc(choiceVal, hidden_val, scale_val, encode_val)
    #         if hidden_val == 0 or hidden_val == 1:
    #             return render_template(
    #                 "classifier_page.html",
    #                 acc=ret_vals[0],
    #                 report=[
    #                     ret_vals[1].to_html(
    #                         classes=[
    #                             "table",
    #                             "table-bordered",
    #                             "table-striped",
    #                             "table-hover",
    #                             "thead-light",
    #                         ]
    #                     )
    #                 ],
    #                 conf_matrix=[
    #                     ret_vals[2].to_html(
    #                         classes=[
    #                             "table",
    #                             "table-bordered",
    #                             "table-striped",
    #                             "table-hover",
    #                             "thead-light",
    #                         ]
    #                     )
    #                 ],
    #                 choice=hidden_val,
    #                 classifier_used=classifier,
    #                 active="classify",
    #                 title="Classify",
    #                 cols=columns,
    #             )
    #         elif hidden_val == 2:
    #             return render_template(
    #                 "classifier_page.html",
    #                 acc=ret_vals[0],
    #                 report=ret_vals[1],
    #                 conf_matrix=ret_vals[2],
    #                 choice=hidden_val,
    #                 classifier_used=classifier,
    #                 active="classify",
    #                 title="Classify",
    #                 cols=columns,
    #             )

    #     elif classifier == 3:

    #         scale_val = 1
    #         ret_vals = knn.KNearestNeighbours(
    #             choiceVal, hidden_val, scale_val, encode_val
    #         )
    #         if hidden_val == 0 or hidden_val == 1:
    #             return render_template(
    #                 "classifier_page.html",
    #                 acc=ret_vals[0],
    #                 report=[
    #                     ret_vals[1].to_html(
    #                         classes=[
    #                             "table",
    #                             "table-bordered",
    #                             "table-striped",
    #                             "table-hover",
    #                             "thead-light",
    #                         ]
    #                     )
    #                 ],
    #                 conf_matrix=[
    #                     ret_vals[2].to_html(
    #                         classes=[
    #                             "table",
    #                             "table-bordered",
    #                             "table-striped",
    #                             "table-hover",
    #                             "thead-light",
    #                         ]
    #                     )
    #                 ],
    #                 choice=hidden_val,
    #                 classifier_used=classifier,
    #                 active="classify",
    #                 title="Classify",
    #                 cols=columns,
    #             )
    #         elif hidden_val == 2:
    #             return render_template(
    #                 "classifier_page.html",
    #                 acc=ret_vals[0],
    #                 report=ret_vals[1],
    #                 conf_matrix=ret_vals[2],
    #                 choice=hidden_val,
    #                 classifier_used=classifier,
    #                 active="classify",
    #                 title="Classify",
    #                 cols=columns,
    #             )

    #     elif classifier == 4:
    #         ret_vals = dtree.DecisionTree(choiceVal, hidden_val, scale_val, encode_val)
    #         if hidden_val == 0 or hidden_val == 1:
    #             return render_template(
    #                 "classifier_page.html",
    #                 acc=ret_vals[0],
    #                 report=[
    #                     ret_vals[1].to_html(
    #                         classes=[
    #                             "table",
    #                             "table-bordered",
    #                             "table-striped",
    #                             "table-hover",
    #                             "thead-light",
    #                         ]
    #                     )
    #                 ],
    #                 conf_matrix=[
    #                     ret_vals[2].to_html(
    #                         classes=[
    #                             "table",
    #                             "table-bordered",
    #                             "table-striped",
    #                             "table-hover",
    #                             "thead-light",
    #                         ]
    #                     )
    #                 ],
    #                 choice=hidden_val,
    #                 classifier_used=classifier,
    #                 active="classify",
    #                 title="Classify",
    #                 cols=columns,
    #             )
    #         elif hidden_val == 2:
    #             return render_template(
    #                 "classifier_page.html",
    #                 acc=ret_vals[0],
    #                 report=ret_vals[1],
    #                 conf_matrix=ret_vals[2],
    #                 choice=hidden_val,
    #                 classifier_used=classifier,
    #                 active="classify",
    #                 title="Classify",
    #                 cols=columns,
    #             )
    #     elif classifier == 5:
    #         ret_vals = rfc.RandomForest(choiceVal, hidden_val, scale_val, encode_val)
    #         if hidden_val == 0 or hidden_val == 1:
    #             return render_template(
    #                 "classifier_page.html",
    #                 acc=ret_vals[0],
    #                 report=[
    #                     ret_vals[1].to_html(
    #                         classes=[
    #                             "table",
    #                             "table-bordered",
    #                             "table-striped",
    #                             "table-hover",
    #                             "thead-light",
    #                         ]
    #                     )
    #                 ],
    #                 conf_matrix=[
    #                     ret_vals[2].to_html(
    #                         classes=[
    #                             "table",
    #                             "table-bordered",
    #                             "table-striped",
    #                             "table-hover",
    #                             "thead-light",
    #                         ]
    #                     )
    #                 ],
    #                 choice=hidden_val,
    #                 classifier_used=classifier,
    #                 active="classify",
    #                 title="Classify",
    #                 cols=columns,
    #             )
    #         elif hidden_val == 2:
    #             return render_template(
    #                 "classifier_page.html",
    #                 acc=ret_vals[0],
    #                 report=ret_vals[1],
    #                 conf_matrix=ret_vals[2],
    #                 choice=hidden_val,
    #                 classifier_used=classifier,
    #                 active="classify",
    #                 title="Classify",
    #                 cols=columns,
    #             )
        elif request.method == "GET":
         columns = vis.get_columns()
         return render_template(
            "classifier_page.html", active="classify", title="Classify", cols=columns
         )


@app.route("/clear", methods=["GET"])
def clear():
    session.clear()
    return redirect("/")


@app.route("/visualize", methods=["GET", "POST"])
@nocache
def visualize():
    if request.method == "POST":
        
        x_col = request.form.getlist('x_col')
        y_col = request.form["y_col"]
        b_col = request.form["b_col"]
        t_col = request.form["t_col"]
        context,html_code,s,r2yz_dx,r2dz_x = causalg.causal_graph(y_col,x_col,b_col,t_col)
        df = vis.xy_plot(x_col, y_col)
        heights = np.array(df[x_col]).tolist()
        weights = np.array(df[y_col]).tolist()

        newlist = []
        for h, w in zip(heights, weights):
            newlist.append({"x": h, "y": w})
        ugly_blob = str(newlist).replace("'", "")

        columns = vis.get_columns()
        return render_template(
            "visualize.html",
            nodes = context['nodes'],
            edges = context['edges'],
            htmlc = html_code,
            estimate =  s['estimate'],
            se = s['se'],
            t_statistic = s['t_statistic'],
            r2yd_x = round(s['r2yd_x']*100,1),
            rv_q = round(s['rv_q']*100,1),
            rv_qa = round(s['rv_qa']*100,1),
            f2yd_x = round(s['f2yd_x']*100,1),
            dof = s['dof'],
            r2yzdx = round(r2yz_dx,1),
            r2dzx = round(r2dz_x,1),
            cols = columns,
            src="img/graph.png",
            xy_src="img/graph.png",
            posted=1,
            data=ugly_blob,
            active="visualize",
            x_col_name=str(x_col),
            y_col_name=str(y_col),
            title="Visualize",
        )

    else:
        # vis.pair_plot()
        columns = vis.get_columns()
        return render_template(
            "visualize.html",
            cols=columns,
            src="img/pairplot1.png",
            posted=0,
            active="visualize",
            title="Visualize",
        )


@app.route("/col.csv")
@nocache
def col():
    return send_file("visualization/col.csv", mimetype="text/csv", as_attachment=True)


@app.route("/pairplot1.png")
@nocache
def pairplot1():
    return send_file(
        "static/img/pairplot1.png", mimetype="image/png", as_attachment=True
    )


@app.route("/tree.png")
@nocache
def tree():
    return send_file("static/img/tree.png", mimetype="image/png", as_attachment=True)
