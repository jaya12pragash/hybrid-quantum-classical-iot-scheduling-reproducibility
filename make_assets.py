"""Generate manuscript figures, tables, and claim macros from archived CSV logs."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIG = ROOT / "figures"
TABLE = ROOT / "tables"
FIG.mkdir(exist_ok=True)
TABLE.mkdir(exist_ok=True)

def load(name):
    with (DATA / name).open(newline="") as f:
        return list(csv.DictReader(f))

seed = load("per_seed_results.csv")
classes = load("per_class_results.csv")
outcomes = ("TIMELY", "LATE", "DROPPED_EXPIRY", "DROPPED_SHED", "PENDING")
policies = [
    "EDF", "LLF", "Threshold Best-Effort Shedding",
    "Matched MLP (0 ms modeled delay)", "Matched MLP (15 ms modeled delay)",
    "Guarded MLP (15 ms nonurgent delay)", "Trained VQC (0 ms modeled delay)",
    "Trained VQC (15 ms modeled delay)", "Guarded VQC (15 ms nonurgent delay)",
]
labels = ["EDF", "LLF", "Threshold shedding", "MLP, 0 ms", "MLP, 15 ms",
          "Guarded MLP", "VQC, 0 ms", "VQC, 15 ms", "Guarded VQC"]
by_policy = defaultdict(lambda: defaultdict(int))
by_class = defaultdict(lambda: defaultdict(int))
for r in seed:
    p = r["policy"]
    for key in ("total", *outcomes): by_policy[p][key] += int(r[key])
    assert sum(int(r[k]) for k in outcomes) == int(r["total"])
for r in classes:
    key = (r["policy"], int(r["class"]))
    for field in ("total", *outcomes): by_class[key][field] += int(r[field])
    assert sum(int(r[k]) for k in outcomes) == int(r["total"])
assert len(seed) == 27 and len(classes) == 81
assert set(int(r["seed"]) for r in seed) == {777, 888, 999}
assert all(by_policy[p]["total"] == 4556 for p in policies)
manifest=json.loads((DATA/"run_manifest.json").read_text())
feasibility=json.loads((DATA/"sla_feasibility.json").read_text())
assert manifest["train_seeds"] == [101,202] and manifest["test_seeds"] == [777,888,999]
assert manifest["training_experiences"] == 233 and manifest["model_trainable_parameters"] == 18
for p in policies:
    for field in ("total", *outcomes):
        assert by_policy[p][field] == sum(by_class[p, c][field] for c in range(3))
    assert [by_class[p,c]["total"] for c in range(3)] == [941,2249,1366]

def line(row): return " & ".join(row) + r" \\" + "\n"
main = [r"\begin{table*}[t]", r"\caption{Held-out packet outcomes, three paired traces (4,556 packets per policy). The 15 ms values are modeled event-time delays.}",
        r"\label{tab:main}", r"\centering\footnotesize", r"\begin{tabular}{lrrrrrr}", r"\toprule",
        line(["Policy", "Timely", "Late", "Expiry", "Shed", "Pending", "Class 0 timely"]), r"\midrule"]
for p,l in zip(policies,labels):
    m=by_policy[p]
    main.append(line([l,*[f"{m[k]:,}" for k in outcomes],f"{by_class[p,0]['TIMELY']:,}/941"]).rstrip())
main += [r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
(TABLE/"main_results.tex").write_text("\n".join(main)+"\n")

chosen = [policies[i] for i in (0,2,5,7,8)]
short = [labels[i] for i in (0,2,5,7,8)]
detail = [r"\begin{table*}[t]",r"\caption{Class-resolved outcomes on the same held-out traces; every row conserves its class packet total.}",r"\label{tab:class}",r"\centering\footnotesize",
          r"\begin{tabular}{llrrrrrr}",r"\toprule",
          line(["Policy","Class","Generated","Timely","Late","Expiry","Shed","Pending"]),r"\midrule"]
for p,l in zip(chosen,short):
    for c in range(3):
        m=by_class[p,c]
        detail.append(line([l if c==0 else "",str(c),f"{m['total']:,}",*[f"{m[k]:,}" for k in outcomes]]).rstrip())
    detail.append(r"\addlinespace")
detail += [r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
(TABLE/"class_results.tex").write_text("\n".join(detail)+"\n")

paired=[r"\begin{table}[H]",r"\caption{Paired held-out differences in timely deliveries. One trace seed, not one packet, is the replicate.}",r"\label{tab:paired}",r"\centering\footnotesize",r"\begin{tabular}{rrrrr}",r"\toprule",
        line(["Seed","Packets","Guard MLP","Guard VQC","Difference"]),r"\midrule"]
differences=[]
for s in (777,888,999):
    rowmap={r["policy"]:r for r in seed if int(r["seed"])==s}
    m=int(rowmap[policies[5]]["TIMELY"]); v=int(rowmap[policies[8]]["TIMELY"])
    differences.append(v-m)
    paired.append(line([str(s),str(rowmap[policies[0]]["total"]),str(m),str(v),f"{v-m:+d}"]).rstrip())
paired += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
assert differences==[18,17,11]
(TABLE/"paired_results.tex").write_text("\n".join(paired)+"\n")

timing_policy=[policies[i] for i in (3,4,5,6,7,8)]
timing_labels=[labels[i] for i in (3,4,5,6,7,8)]
timing=[r"\begin{table*}[t]",r"\caption{Decision-time ablation. Wall time is the archived host measurement of the complete policy callback; it is distinct from modeled event delay and includes the fast branch in guarded rows.}",r"\label{tab:timing}",r"\centering\footnotesize",r"\begin{tabular}{lrrrrr}",r"\toprule",
        line(["Policy","Decisions","Modeled nonurgent delay (ms)","Mean local callback (ms)","Link busy (\\%)","Timely"]),r"\midrule"]
busy_values=[]
for p,l in zip(timing_policy,timing_labels):
    rs=[r for r in seed if r["policy"]==p]
    decisions=sum(int(r["decisions"]) for r in rs)
    wall=sum(float(r["mean_local_scorer_ms"])*int(r["decisions"]) for r in rs)/decisions
    busy=100*sum(int(r["link_busy_us"]) for r in rs)/(3*5_000_000)
    busy_values.append(busy)
    timing.append(line([l,f"{decisions:,}","0" if "0 ms" in p else "15",f"{wall:.3f}",f"{busy:.2f}",f"{by_policy[p]['TIMELY']:,}"]).rstrip())
timing += [r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
(TABLE/"timing_results.tex").write_text("\n".join(timing)+"\n")

ablation=[r"\begin{table*}[t]",r"\caption{Architectural ablations on the same held-out traces. Numeric pairs are baseline/intervention. The guard changes urgent selection, modeled delay and cancellation jointly.}",r"\label{tab:architecture_ablation}",r"\centering\footnotesize",r"\begin{tabular}{p{.25\textwidth}rrp{.34\textwidth}}",r"\toprule",
          line(["Controlled comparison","Class 0 timely","All timely","Attribution boundary"]),r"\midrule"]
def paired_ablation(label,old,new,note):
    return line([label,f"{by_class[old,0]['TIMELY']:,} / {by_class[new,0]['TIMELY']:,}",
                 f"{by_policy[old]['TIMELY']:,} / {by_policy[new]['TIMELY']:,}",note]).rstrip()
ablation += [
    paired_ablation("MLP: 0 to 15 ms",policies[3],policies[4],"Same trained scorer; modeled serial delay changes."),
    paired_ablation("VQC: 0 to 15 ms",policies[6],policies[7],"Same trained scorer; modeled serial delay changes."),
    paired_ablation("MLP: unguarded to guarded, 15 ms",policies[4],policies[5],"Urgent EDF, urgent zero-delay and cancellation change together."),
    paired_ablation("VQC: unguarded to guarded, 15 ms",policies[7],policies[8],"Same bundled guard; cannot assign gain to a single guard element."),
    paired_ablation("Guarded: MLP to VQC",policies[5],policies[8],"Shared guard and parameter count; architectures and fitted weights differ."),
    r"\bottomrule",r"\end{tabular}",r"\end{table*}"]
(TABLE/"ablation_results.tex").write_text("\n".join(ablation)+"\n")

setup=[r"\begin{table}[t]",r"\caption{Locked experiment protocol; values are configuration, not measured hardware specifications.}",r"\label{tab:setup}",r"\centering\footnotesize",r"\begin{tabular}{lp{.55\columnwidth}}",r"\toprule",
       line(["Setting","Value"]),r"\midrule",
       line(["Time/medium",r"5 s; integer $\mu$s; one nonpreemptive 1 Mbps link"]),
       line(["Arrivals/payload","Exponential, nominal 300 packets/s; 2,000--7,999 bits"]),
       line(["Class/slack","Probabilities 0.2/0.5/0.3; class 0: 10--40 ms; classes 1--2: 50--150 ms"]),
       line(["Training","Seeds 101, 202; 233 transitions; 5 epochs; fixed batch 32"]),
       line(["Evaluation","Seeds 777, 888, 999; identical trace within each seed"]),
       line(["Scorers","9 inputs; 18 parameters each; statevector VQC or tanh MLP"]),
       line(["Delay ablation","0 or 15 ms modeled nonurgent; urgent guard branch 0"]),
       r"\bottomrule",r"\end{tabular}",r"\end{table}"]
(TABLE/"setup.tex").write_text("\n".join(setup)+"\n")

def macro(name,val): return rf"\newcommand{{\{name}}}{{{val}}}" + "\n"
claims = "%% Generated from archived results. Do not edit by hand.\n"
for n,p,k in [("NPackets",policies[0],"total"),("EDFTimely",policies[0],"TIMELY"),
              ("ThresholdTimely",policies[2],"TIMELY"),("GuardMLPTimely",policies[5],"TIMELY"),
              ("GuardVQCTimely",policies[8],"TIMELY"),("SlowVQCTimely",policies[7],"TIMELY")]:
    claims += macro(n, f"{by_policy[p][k]:,}")
for n,p in [("GuardZero",policies[5]),("GuardOne",policies[8]),("SlowZero",policies[4]),("SlowOne",policies[7]),("ThresholdZero",policies[2]),("ZeroVQCUrgent",policies[6]),("ZeroMLPUrgent",policies[3])]:
    claims += macro(n,by_class[p,0]["TIMELY"])
claims += macro("ClassZeroTotal",by_class[policies[0],0]["total"])
claims += macro("GuardRate",f"{100*by_class[policies[8],0]['TIMELY']/941:.2f}")
for name,key in [("IsolatedFeasible","isolated_feasible_with_15ms_each"),
                 ("MinimumUrgent","target_timely_95pct"),
                 ("DelayFeasible","largest_delay_for_95pct_isolated_feasibility_us")]:
    claims += macro(name,f"{feasibility[key]:,}")
claims += macro("TrainingTransitions",manifest["training_experiences"])
claims += macro("GuardedDifference",by_policy[policies[8]]["TIMELY"]-by_policy[policies[5]]["TIMELY"])
for name,p,c in [("VQCClassOne",policies[8],1),("MLPClassOne",policies[5],1),
                 ("VQCClassTwo",policies[8],2),("MLPClassTwo",policies[5],2)]:
    claims += macro(name,by_class[p,c]["TIMELY"])
(TABLE/"claims.tex").write_text(claims)

plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"pdf.fonttype":42,
                     "axes.spines.top":False,"axes.spines.right":False})
colors = ["#4c78a8","#f58518","#54a24b","#b279a2","#e45756"]
fig,ax=plt.subplots(figsize=(7.05,3.25),layout="constrained")
x=np.arange(3); width=.15
for j,(p,l) in enumerate(zip(chosen,short)):
    y=[100*by_class[p,c]["TIMELY"]/by_class[p,c]["total"] for c in range(3)]
    ax.bar(x+(j-2)*width,y,width,label=l,color=colors[j])
ax.set_xticks(x,["0: urgent","1: massive IoT","2: best effort"])
ax.set_ylim(0,100);ax.set_ylabel("Timely / generated in class (%)")
ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
ax.legend(loc="upper center",bbox_to_anchor=(.5,1.28),ncol=3,frameon=False,fontsize=7.5)
fig.savefig(FIG/"class_timely.pdf",bbox_inches="tight");plt.close(fig)

fig,ax=plt.subplots(figsize=(7.05,3.0),layout="constrained")
compare=[policies[i] for i in (3,4,5,6,7,8)]
labs=[labels[i] for i in (3,4,5,6,7,8)]
x=np.arange(len(compare)); bottom=np.zeros(len(compare))
for key,col,name in [("TIMELY","#54a24b","Timely"),("LATE","#f58518","Late"),
                     ("DROPPED_EXPIRY","#e45756","Expired"),
                     ("DROPPED_SHED","#b279a2","Shed"),("PENDING","#bab0ac","Pending")]:
    h=np.array([100*by_policy[p][key]/4556 for p in compare])
    ax.bar(x,h,bottom=bottom,label=name,color=col);bottom+=h
ax.set_xticks(x,labs,rotation=20,ha="right",fontsize=7.3)
ax.set_ylabel("Fraction of generated packets (%)");ax.set_ylim(0,100)
ax.legend(ncol=5,loc="upper center",bbox_to_anchor=(.5,1.2),frameon=False,fontsize=7.5)
fig.savefig(FIG/"outcome_ablation.pdf",bbox_inches="tight");plt.close(fig)

fig,ax=plt.subplots(figsize=(7.05,2.75),layout="constrained")
seedmap={(r["policy"],int(r["seed"]),int(r["class"])):r for r in classes}
for p,l,col,mark in [(policies[2],"Threshold shedding",colors[1],"o"),
                     (policies[5],"Guarded MLP",colors[2],"s"),
                     (policies[8],"Guarded VQC",colors[4],"^")]:
    y=[100*int(seedmap[p,s,0]["TIMELY"])/int(seedmap[p,s,0]["total"]) for s in (777,888,999)]
    ax.plot([777,888,999],y,marker=mark,label=l,color=col,linewidth=1.5)
ax.axhline(95,color="black",linestyle="--",linewidth=1,label="95% target")
ax.set_xticks([777,888,999]);ax.set_xlabel("Held-out trace seed")
ax.set_ylabel("Class 0 timely / generated (%)");ax.set_ylim(0,102)
ax.legend(ncol=2,loc="upper center",bbox_to_anchor=(.5,1.32),frameon=False,fontsize=7.5)
fig.savefig(FIG/"seed_sla.pdf",bbox_inches="tight");plt.close(fig)
from make_structural_figures import build as build_structural_figures
build_structural_figures(FIG)

mlp_loss=load("mlp_training_loss.csv");vqc_loss=load("training_loss.csv")
fig,axes=plt.subplots(1,2,figsize=(7.05,2.8),sharex=True,layout="constrained")
for ax,rows,label,col in [(axes[0],mlp_loss,"MLP","#4c78a8"),(axes[1],vqc_loss,"VQC","#e45756")]:
    epochs=[int(r["epoch"]) for r in rows]
    before=[float(r["frozen_target_loss_before"]) for r in rows]
    after=[float(r["frozen_target_loss_after"]) for r in rows]
    ax.plot(epochs,before,marker="o",label="Before update",color=col,alpha=.5)
    ax.plot(epochs,after,marker="s",label="After update",color=col)
    ax.set_xticks(epochs);ax.set_xlabel("Epoch");ax.set_title(label)
    ax.grid(alpha=.2);ax.legend(fontsize=7,frameon=False)
axes[0].set_ylabel("Frozen-target batch MSE")
fig.savefig(FIG/"training_loss.pdf",bbox_inches="tight");plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(3.45,3.55),sharex=True,layout="constrained")
for ax,rows,label,col in [(axes[0],mlp_loss,"MLP","#4c78a8"),(axes[1],vqc_loss,"VQC","#e45756")]:
    epochs=[int(r["epoch"]) for r in rows]
    ax.plot(epochs,[float(r["frozen_target_loss_before"]) for r in rows],
            marker="o",label="Before",color=col,alpha=.55)
    ax.plot(epochs,[float(r["frozen_target_loss_after"]) for r in rows],
            marker="s",label="After",color=col)
    ax.set_title(label,fontsize=9);ax.set_xticks(epochs);ax.grid(alpha=.2)
    ax.legend(fontsize=7,frameon=False,ncol=2)
axes[0].set_ylabel("Frozen-target MSE")
axes[1].set_ylabel("Frozen-target MSE");axes[1].set_xlabel("Epoch")
fig.savefig(FIG/"training_loss_column.pdf",bbox_inches="tight");plt.close(fig)
fig,ax=plt.subplots(figsize=(7.05,3.2),layout="constrained")
markers=["o","s","^","D","P","X"]
for i,(p,l,busy) in enumerate(zip(timing_policy,timing_labels,busy_values)):
    timely=100*by_policy[p]["TIMELY"]/4556
    ax.scatter(busy,timely,s=70,marker=markers[i],color=["#4c78a8","#4c78a8","#54a24b","#e45756","#e45756","#b279a2"][i],label=l)
ax.set_xlim(0,105);ax.set_ylim(0,75)
ax.set_xlabel("Link busy fraction across three five-second traces (%)")
ax.set_ylabel("Timely / generated packets (%)")
ax.grid(alpha=.2)
ax.legend(ncol=3,loc="upper center",bbox_to_anchor=(.5,1.25),fontsize=7.2,frameon=False)
fig.savefig(FIG/"timing_tradeoff.pdf",bbox_inches="tight");plt.close(fig)
print("Generated tables and vector figures from 27 policy runs; all conservation checks passed.")
