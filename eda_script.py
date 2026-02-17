import pandas as pd
import numpy as np
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import warnings, os, time
warnings.filterwarnings('ignore')

os.makedirs('/home/claude/eda_output/plots', exist_ok=True)
sns.set_theme(style='whitegrid')
plt.rcParams.update({'font.family':'DejaVu Sans','figure.dpi':120,
                     'axes.titlesize':13,'axes.labelsize':11})

# ─────────────────────────────────────────────
# STEP 1 — Load CSVs into SQLite
# ─────────────────────────────────────────────
print("="*60)
print("STEP 1: Loading CSVs → SQLite Database")
print("="*60)
conn = sqlite3.connect('/home/claude/inventory.db')
csv_folder = '/home/claude/vendor_data'
table_map = {
    'purchases.csv':'purchases', 'purchase_prices.csv':'purchase_prices',
    'vendor_invoice.csv':'vendor_invoice', 'sales.csv':'sales',
    'BegInvFINAL12312016.csv':'begin_inventory',
    'EndInvFINAL12312016.csv':'end_inventory',
}
for fname, tname in table_map.items():
    df_tmp = pd.read_csv(f'{csv_folder}/{fname}')
    df_tmp.to_sql(tname, conn, if_exists='replace', index=False)
    print(f"  ✓ {tname:25s} → {len(df_tmp):,} rows | {len(df_tmp.columns)} cols")

# ─────────────────────────────────────────────
# STEP 2 — SQL EDA
# ─────────────────────────────────────────────
print("\n"+"="*60)
print("STEP 2: SQL — Record Counts")
print("="*60)
tables = ['purchases','purchase_prices','vendor_invoice','sales','begin_inventory','end_inventory']
for t in tables:
    n = pd.read_sql(f"SELECT COUNT(*) AS cnt FROM {t}", conn).iloc[0,0]
    print(f"  {t:25s}: {n:>8,} records")

print("\n"+"="*60)
print("STEP 3: SQL — Top 5 Sample Rows per Table")
print("="*60)
for t in tables:
    print(f"\n{'─'*55}")
    print(f"  TABLE: {t.upper()}")
    print('─'*55)
    sample = pd.read_sql(f"SELECT * FROM {t} LIMIT 5", conn)
    print(sample.to_string(index=False))

# ─────────────────────────────────────────────
# STEP 4 — SQL Summary Queries
# ─────────────────────────────────────────────
print("\n"+"="*60)
print("STEP 4: SQL — Purchase Summary")
print("="*60)
# Deduplicate purchase_prices per brand (take avg price)
pur_sum = pd.read_sql("""
    SELECT p.VendorNumber, p.VendorName, p.Brand, p.Description,
           AVG(pp.Price)         AS actual_price,
           SUM(p.PurchasePrice)  AS purchase_price,
           SUM(p.Quantity)       AS total_purchase_qty,
           SUM(p.Dollars)        AS total_purchase_dollars
    FROM purchases p
    LEFT JOIN (
        SELECT Brand, AVG(Price) AS Price, AVG(Volume) AS Volume
        FROM purchase_prices GROUP BY Brand
    ) pp ON p.Brand = pp.Brand
    WHERE p.PurchasePrice > 0
    GROUP BY p.VendorNumber, p.VendorName, p.Brand, p.Description
    ORDER BY total_purchase_dollars DESC
""", conn)
print(f"  Shape: {pur_sum.shape}")
print(pur_sum.head(10).to_string(index=False))

print("\n"+"="*60)
print("STEP 4b: SQL — Sales Summary")
print("="*60)
sal_sum = pd.read_sql("""
    SELECT VendorNumber, VendorName, Brand, Description,
           SUM(SalesQuantity) AS total_sales_qty,
           SUM(SalesDollars)  AS total_sales_dollars,
           AVG(SalesPrice)    AS avg_sales_price,
           SUM(ExciseTax)     AS total_excise_tax
    FROM sales
    GROUP BY VendorNumber, Brand, Description
    ORDER BY total_sales_dollars DESC
""", conn)
print(f"  Shape: {sal_sum.shape}")
print(sal_sum.head(10).to_string(index=False))

print("\n"+"="*60)
print("STEP 4c: SQL — Freight Summary")
print("="*60)
freight = pd.read_sql("""
    SELECT VendorNumber, VendorName,
           ROUND(SUM(Freight),2) AS freight_cost
    FROM vendor_invoice
    GROUP BY VendorNumber, VendorName
    ORDER BY freight_cost DESC
""", conn)
print(freight.head(10).to_string(index=False))

# ─────────────────────────────────────────────
# STEP 5 — Final Aggregated Table
# ─────────────────────────────────────────────
print("\n"+"="*60)
print("STEP 5: Creating Final Aggregated Table")
print("="*60)
t0 = time.time()

# Merge with pandas (avoid SQL join duplication)
final = pur_sum.merge(
    sal_sum[['VendorNumber','Brand','total_sales_qty','total_sales_dollars','avg_sales_price','total_excise_tax']],
    on=['VendorNumber','Brand'], how='left'
).merge(
    freight[['VendorNumber','freight_cost']],
    on='VendorNumber', how='left'
)
final.fillna(0, inplace=True)
for col in ['VendorName','Description']:
    final[col] = final[col].astype(str).str.strip()

print(f"  Final table shape  : {final.shape}")
print(f"  Time taken         : {time.time()-t0:.2f}s")
print(f"  Unique Vendors     : {final['VendorNumber'].nunique()}")
print(f"  Unique Brands      : {final['Brand'].nunique()}")
print(final.head(6).to_string(index=False))

# ─────────────────────────────────────────────
# STEP 6 — Data Cleaning
# ─────────────────────────────────────────────
print("\n"+"="*60)
print("STEP 6: Data Cleaning")
print("="*60)
print(f"  Missing values:\n{final.isnull().sum().to_string()}")
print(f"\n  Data types:\n{final.dtypes.to_string()}")
print(f"\n  Duplicates: {final.duplicated().sum()}")

# ─────────────────────────────────────────────
# STEP 7 — Feature Engineering
# ─────────────────────────────────────────────
print("\n"+"="*60)
print("STEP 7: Feature Engineering")
print("="*60)
final['gross_profit']      = final['total_sales_dollars'] - final['total_purchase_dollars']
final['profit_margin']     = np.where(final['total_sales_dollars']>0,
    (final['gross_profit']/final['total_sales_dollars'])*100, 0)
final['stock_turnover']    = np.where(final['total_purchase_qty']>0,
    final['total_sales_qty']/final['total_purchase_qty'], 0)
final['sales_to_pur_ratio']= np.where(final['total_purchase_dollars']>0,
    final['total_sales_dollars']/final['total_purchase_dollars'], 0)

# Save to DB
final.to_sql('vendor_sales_summary', conn, if_exists='replace', index=False)
print("  ✓ vendor_sales_summary saved to DB")
print(f"\n  Sample (gross_profit check):")
print(final[['VendorName','Description','total_purchase_dollars','total_sales_dollars',
             'gross_profit','profit_margin','stock_turnover']].head(10).to_string(index=False))

# Filter
df = final[(final['gross_profit']>0)&(final['profit_margin']>0)&(final['total_sales_qty']>0)].copy()
print(f"\n  Records after filter: {len(df):,} / {len(final):,}")

# ─────────────────────────────────────────────
# STEP 8 — Summary Statistics
# ─────────────────────────────────────────────
print("\n"+"="*60)
print("STEP 8: Summary Statistics")
print("="*60)
num_cols = ['purchase_price','actual_price','total_purchase_qty','total_purchase_dollars',
            'total_sales_qty','total_sales_dollars','gross_profit','profit_margin',
            'stock_turnover','freight_cost']
print(df[num_cols].describe().T.round(2).to_string())

# ─────────── PLOTS ───────────
def fmt(v):
    if v>=1e6: return f'${v/1e6:.2f}M'
    if v>=1e3: return f'${v/1e3:.1f}K'
    return f'${v:.0f}'

PAL = '#2E75B6'

# PLOT 1 — Histograms
print("\n[PLOT 1] Histograms...")
fig, axes = plt.subplots(3,4,figsize=(18,12))
plot_cols=['purchase_price','actual_price','total_purchase_qty','total_purchase_dollars',
           'total_sales_qty','total_sales_dollars','gross_profit','profit_margin',
           'stock_turnover','freight_cost','sales_to_pur_ratio','avg_sales_price']
for i,col in enumerate(plot_cols):
    ax=axes.flatten()[i]
    d=df[col].clip(upper=df[col].quantile(0.98))
    ax.hist(d, bins=35, color=PAL, edgecolor='white', linewidth=0.4)
    ax.set_title(col.replace('_',' ').title(), fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'{int(x):,}'))
fig.suptitle('Distribution of All Numerical Features', fontsize=15, fontweight='bold', y=1.01)
plt.tight_layout(); plt.savefig('/home/claude/eda_output/plots/01_histograms.png',bbox_inches='tight'); plt.close()
print("  ✓ 01_histograms.png")

# PLOT 2 — Box Plots
print("[PLOT 2] Box Plots...")
fig, axes = plt.subplots(3,4,figsize=(18,12))
for i,col in enumerate(plot_cols):
    ax=axes.flatten()[i]
    d=df[col].clip(upper=df[col].quantile(0.98))
    ax.boxplot(d, patch_artist=True,
               boxprops=dict(facecolor='#BDD7EE',color='#1F4E79'),
               medianprops=dict(color='#C00000',linewidth=2),
               whiskerprops=dict(color='#1F4E79'), capprops=dict(color='#1F4E79'),
               flierprops=dict(marker='o',color='#FF4444',alpha=0.3,markersize=3))
    ax.set_title(col.replace('_',' ').title(), fontsize=10); ax.set_xticks([])
fig.suptitle('Box Plots — Outlier Detection', fontsize=15, fontweight='bold')
plt.tight_layout(); plt.savefig('/home/claude/eda_output/plots/02_boxplots.png',bbox_inches='tight'); plt.close()
print("  ✓ 02_boxplots.png")

# PLOT 3 — Count Plots
print("[PLOT 3] Count Plots...")
fig, axes = plt.subplots(1,2,figsize=(16,6))
tv=df['VendorName'].value_counts().head(10); tb=df['Description'].value_counts().head(10)
axes[0].barh(tv.index[::-1], tv.values[::-1], color=sns.color_palette('Blues_d',10))
axes[0].set_title('Top 10 Vendors by Records'); axes[0].set_xlabel('Count')
axes[1].barh(tb.index[::-1], tb.values[::-1], color=sns.color_palette('Blues_d',10))
axes[1].set_title('Top 10 Brands by Records'); axes[1].set_xlabel('Count')
plt.suptitle('Data Distribution — Vendor & Brand Records', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.savefig('/home/claude/eda_output/plots/03_count_plots.png',bbox_inches='tight'); plt.close()
print("  ✓ 03_count_plots.png")

# PLOT 4 — Correlation Heatmap
print("[PLOT 4] Correlation Heatmap...")
fig, ax = plt.subplots(figsize=(12,8))
corr=df[num_cols].corr()
mask=np.triu(np.ones_like(corr,dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu',
            linewidths=0.5, ax=ax, annot_kws={'size':8}, vmin=-1, vmax=1, center=0)
ax.set_title('Correlation Heatmap — Numerical Features', fontweight='bold', pad=15)
plt.tight_layout(); plt.savefig('/home/claude/eda_output/plots/04_correlation_heatmap.png',bbox_inches='tight'); plt.close()
print("  ✓ 04_correlation_heatmap.png")

# PLOT 5 — Research Q1: Target Brands
print("[PLOT 5] Target Brands Scatter...")
bp = df.groupby('Description').agg(total_sales_dollars=('total_sales_dollars','sum'),
                                    profit_margin=('profit_margin','mean')).reset_index()
ls_thresh = bp['total_sales_dollars'].quantile(0.15)
hm_thresh = bp['profit_margin'].quantile(0.85)
target = bp[(bp['total_sales_dollars']<ls_thresh)&(bp['profit_margin']>hm_thresh)]
print(f"  Target Brands: {len(target)}  |  LowSales<{ls_thresh:.0f}  HighMargin>{hm_thresh:.1f}%")

fig, ax = plt.subplots(figsize=(11,7))
filt = bp[bp['total_sales_dollars']<bp['total_sales_dollars'].quantile(0.95)]
nt = filt[~filt['Description'].isin(target['Description'])]
tg = filt[filt['Description'].isin(target['Description'])]
ax.scatter(nt['total_sales_dollars'], nt['profit_margin'], alpha=0.3, color='#2E75B6', s=40, label='Normal Brands')
ax.scatter(tg['total_sales_dollars'], tg['profit_margin'], color='#C00000', s=80, alpha=0.85, label=f'Target Brands ({len(target)})', zorder=5)
ax.axvline(ls_thresh, color='orange', linestyle='--', lw=1.8, label=f'Low Sales Threshold (${ls_thresh:,.0f})')
ax.axhline(hm_thresh, color='green',  linestyle='--', lw=1.8, label=f'High Margin Threshold ({hm_thresh:.1f}%)')
ax.set_xlabel('Total Sales ($)'); ax.set_ylabel('Profit Margin (%)')
ax.set_title('Brands Needing Promotional / Pricing Adjustments', fontweight='bold')
ax.legend(fontsize=9); ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'${x:,.0f}'))
plt.tight_layout(); plt.savefig('/home/claude/eda_output/plots/05_target_brands.png',bbox_inches='tight'); plt.close()
print("  ✓ 05_target_brands.png")

# PLOT 6 — Top 10 Vendors & Brands by Sales
print("[PLOT 6] Top Vendors & Brands by Sales...")
tv_sales = df.groupby('VendorName')['total_sales_dollars'].sum().nlargest(10)
tb_sales = df.groupby('Description')['total_sales_dollars'].sum().nlargest(10)
print(f"\n  Top 10 Vendors by Sales:")
for k,v in tv_sales.items(): print(f"    {k:30s}: {fmt(v)}")
print(f"\n  Top 10 Brands by Sales:")
for k,v in tb_sales.items(): print(f"    {k:30s}: {fmt(v)}")

fig, axes = plt.subplots(1,2,figsize=(18,7))
c10=sns.color_palette('Blues_d',10)
b1=axes[0].barh(tv_sales.index[::-1], tv_sales.values[::-1], color=c10)
for bar,val in zip(b1, tv_sales.values[::-1]):
    axes[0].text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2, fmt(val), va='center', fontsize=8, fontweight='bold')
axes[0].set_title('Top 10 Vendors by Total Sales', fontweight='bold')
axes[0].set_xlabel('Sales ($)'); axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'${x/1e3:.0f}K'))
axes[0].set_xlim(0, tv_sales.max()*1.22)

b2=axes[1].barh(tb_sales.index[::-1], tb_sales.values[::-1], color=c10)
for bar,val in zip(b2, tb_sales.values[::-1]):
    axes[1].text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2, fmt(val), va='center', fontsize=8, fontweight='bold')
axes[1].set_title('Top 10 Brands by Total Sales', fontweight='bold')
axes[1].set_xlabel('Sales ($)'); axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'${x/1e3:.0f}K'))
axes[1].set_xlim(0, tb_sales.max()*1.22)

plt.suptitle('Top 10 Vendors & Brands — Sales Performance', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('/home/claude/eda_output/plots/06_top_vendors_brands.png',bbox_inches='tight'); plt.close()
print("  ✓ 06_top_vendors_brands.png")

# PLOT 7 — Pareto (Purchase Contribution)
print("[PLOT 7] Purchase Contribution Pareto...")
vp = df.groupby('VendorName').agg(total_purchase_dollars=('total_purchase_dollars','sum')).reset_index()
vp['pct'] = (vp['total_purchase_dollars']/vp['total_purchase_dollars'].sum()*100).round(2)
vp = vp.sort_values('pct', ascending=False)
top10c = vp.head(10)
cumsum = top10c['pct'].cumsum()
top10_total = top10c['pct'].sum()
print(f"  Top 10 vendors purchase contribution: {top10_total:.2f}%")
print(top10c[['VendorName','pct','total_purchase_dollars']].to_string(index=False))

fig, ax1 = plt.subplots(figsize=(14,6))
c_pareto=sns.color_palette('Blues_d',10)
bars=ax1.bar(range(len(top10c)), top10c['pct'], color=c_pareto)
ax1.set_xticks(range(len(top10c))); ax1.set_xticklabels(top10c['VendorName'], rotation=30, ha='right', fontsize=9)
ax1.set_ylabel('Purchase Contribution (%)'); ax1.set_title('Vendor Purchase Contribution — Pareto Chart', fontweight='bold')
ax2=ax1.twinx()
ax2.plot(range(len(top10c)), cumsum, color='#C00000', marker='o', lw=2, markersize=5)
ax2.axhline(80, color='gray', linestyle='--', alpha=0.5); ax2.set_ylabel('Cumulative %', color='#C00000'); ax2.set_ylim(0,110)
for bar,val in zip(bars, top10c['pct']):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f'{val:.1f}%', ha='center', fontsize=8, fontweight='bold')
plt.tight_layout(); plt.savefig('/home/claude/eda_output/plots/07_purchase_pareto.png',bbox_inches='tight'); plt.close()
print("  ✓ 07_purchase_pareto.png")

# PLOT 8 — Donut Chart
print("[PLOT 8] Donut Chart...")
fig, ax = plt.subplots(figsize=(9,7))
labels_d = list(top10c['VendorName'])+['Other Vendors']
sizes_d  = list(top10c['pct'])+[100-top10_total]
colors_d = sns.color_palette('Blues',len(labels_d))
wedges,_,at = ax.pie(sizes_d, labels=None, autopct='%1.1f%%', startangle=140,
                     colors=colors_d, pctdistance=0.82, wedgeprops=dict(width=0.5))
for a in at: a.set_fontsize(7.5)
ax.legend(wedges, labels_d, loc='center left', bbox_to_anchor=(1,0.5), fontsize=8)
ax.set_title('Purchase Contribution: Top 10 Vendors vs Others', fontweight='bold')
plt.tight_layout(); plt.savefig('/home/claude/eda_output/plots/08_donut.png',bbox_inches='tight'); plt.close()
print("  ✓ 08_donut.png")

# PLOT 9 — Bulk Purchasing
print("[PLOT 9] Bulk Purchasing Impact...")
df2=df.copy()
df2['unit_price'] = df2['total_purchase_dollars']/df2['total_purchase_qty'].replace(0,np.nan)
df2['order_size'] = pd.qcut(df2['total_purchase_qty'], q=3, labels=['Small','Medium','Large'])
grp = df2.groupby('order_size')['unit_price'].mean().round(2)
pct_r = (grp['Small']-grp['Large'])/grp['Small']*100
print(f"  Unit Prices — Small: ${grp['Small']:.2f} | Medium: ${grp['Medium']:.2f} | Large: ${grp['Large']:.2f}")
print(f"  Price Reduction Small→Large: {pct_r:.1f}%")

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6))
c3=['#BDD7EE','#2E75B6','#1F4E79']
bars=ax1.bar(grp.index, grp.values, color=c3, width=0.5, edgecolor='white')
for bar,val in zip(bars, grp.values):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1, f'${val:.2f}', ha='center', fontweight='bold')
ax1.set_title('Avg Unit Price by Order Size', fontweight='bold'); ax1.set_ylabel('Avg Unit Price ($)'); ax1.set_ylim(0, grp.max()*1.2)
bp_data=[df2[df2['order_size']==s]['unit_price'].dropna().clip(upper=df2['unit_price'].quantile(0.95)) for s in ['Small','Medium','Large']]
bp=ax2.boxplot(bp_data, patch_artist=True, labels=['Small','Medium','Large'], medianprops=dict(color='#C00000',lw=2))
for p,c in zip(bp['boxes'],c3): p.set_facecolor(c)
ax2.set_title('Unit Price Distribution by Order Size', fontweight='bold'); ax2.set_ylabel('Unit Price ($)')
plt.suptitle('Impact of Bulk Purchasing on Unit Cost', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('/home/claude/eda_output/plots/09_bulk_purchasing.png',bbox_inches='tight'); plt.close()
print("  ✓ 09_bulk_purchasing.png")

# PLOT 10 — Low Inventory Turnover
print("[PLOT 10] Low Inventory Turnover...")
low_t = df[df['stock_turnover']<1].groupby('VendorName')['stock_turnover'].mean().sort_values().head(10)
print(f"  Top 10 Low Turnover Vendors:")
print(low_t.round(3).to_string())

fig, ax = plt.subplots(figsize=(10,6))
c_red=sns.color_palette('Reds_r',10)
blt=ax.barh(low_t.index[::-1], low_t.values[::-1], color=c_red[::-1])
ax.axvline(0.5, color='orange', linestyle='--', lw=1.5, label='Threshold: 0.5')
for bar,val in zip(blt, low_t.values[::-1]):
    ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
ax.set_title('Top 10 Vendors — Lowest Inventory Turnover', fontweight='bold'); ax.set_xlabel('Avg Stock Turnover'); ax.legend()
plt.tight_layout(); plt.savefig('/home/claude/eda_output/plots/10_low_turnover.png',bbox_inches='tight'); plt.close()
print("  ✓ 10_low_turnover.png")

# PLOT 11 — Unsold Inventory
print("[PLOT 11] Unsold Inventory Capital...")
df['unsold_val'] = np.maximum((df['total_purchase_qty']-df['total_sales_qty'])*df['actual_price'],0)
total_locked = df['unsold_val'].sum()
unsold_v = df.groupby('VendorName')['unsold_val'].sum().nlargest(10)
print(f"  Total Capital Locked: {fmt(total_locked)}")
print(f"  Top 10 Vendors:")
for k,v in unsold_v.items(): print(f"    {k:30s}: {fmt(v)}")

fig, ax = plt.subplots(figsize=(11,6))
c_org=sns.color_palette('Oranges_d',10)
buv=ax.barh(unsold_v.index[::-1], unsold_v.values[::-1], color=c_org[::-1])
for bar,val in zip(buv, unsold_v.values[::-1]):
    ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2, fmt(val), va='center', fontsize=9, fontweight='bold')
ax.set_title(f'Capital Locked in Unsold Inventory by Vendor\n(Total: {fmt(total_locked)})', fontweight='bold')
ax.set_xlabel('Unsold Inventory Value ($)'); ax.set_xlim(0, unsold_v.max()*1.22)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'${x:,.0f}'))
plt.tight_layout(); plt.savefig('/home/claude/eda_output/plots/11_unsold_inventory.png',bbox_inches='tight'); plt.close()
print("  ✓ 11_unsold_inventory.png")

# PLOT 12 — Confidence Interval
print("[PLOT 12] Confidence Intervals...")
vend_sales = df.groupby('VendorName')['total_sales_dollars'].sum()
top_set = vend_sales[vend_sales >= vend_sales.quantile(0.75)].index
low_set = vend_sales[vend_sales <= vend_sales.quantile(0.25)].index
top_m = df[df['VendorName'].isin(top_set)]['profit_margin']
low_m = df[df['VendorName'].isin(low_set)]['profit_margin']

def ci(data, conf=0.95):
    m=data.mean(); se=stats.sem(data)
    tc=stats.t.ppf((1+conf)/2, df=len(data)-1)
    moe=tc*se; return m, m-moe, m+moe

tm,tl,th = ci(top_m); lm,ll,lh = ci(low_m)
print(f"  Top Vendors 95% CI: [{tl:.2f}%, {th:.2f}%]  Mean: {tm:.2f}%")
print(f"  Low Vendors 95% CI: [{ll:.2f}%, {lh:.2f}%]  Mean: {lm:.2f}%")

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,6))
ax1.hist(top_m.clip(0,100), bins=30, color='#2E75B6', edgecolor='white', alpha=0.85, label='Distribution')
ax1.axvline(tm, color='#1F4E79', lw=2, label=f'Mean: {tm:.2f}%')
ax1.axvspan(tl,th, alpha=0.2, color='#2E75B6', label=f'95% CI [{tl:.2f}–{th:.2f}%]')
ax1.set_title('Top Performing Vendors — Profit Margin', fontweight='bold'); ax1.set_xlabel('Profit Margin (%)'); ax1.legend(fontsize=8)

ax2.hist(low_m.clip(0,100), bins=30, color='#C00000', edgecolor='white', alpha=0.85, label='Distribution')
ax2.axvline(lm, color='#7B0000', lw=2, label=f'Mean: {lm:.2f}%')
ax2.axvspan(ll,lh, alpha=0.2, color='#C00000', label=f'95% CI [{ll:.2f}–{lh:.2f}%]')
ax2.set_title('Low Performing Vendors — Profit Margin', fontweight='bold'); ax2.set_xlabel('Profit Margin (%)'); ax2.legend(fontsize=8)
plt.suptitle('95% Confidence Interval — Profit Margin Comparison', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('/home/claude/eda_output/plots/12_confidence_interval.png',bbox_inches='tight'); plt.close()
print("  ✓ 12_confidence_interval.png")

# Hypothesis Testing
print("\n"+"="*60)
print("RESEARCH Q8: Hypothesis Testing (Welch T-Test)")
print("="*60)
print("  H0: No significant difference in profit margin (top vs low vendors)")
print("  H1: Significant difference EXISTS\n")
t_stat, p_val = stats.ttest_ind(top_m, low_m, equal_var=False)
print(f"  T-Statistic : {t_stat:.4f}")
print(f"  P-Value     : {p_val:.6f}")
if p_val < 0.05:
    print("  ✓ REJECT H0 — Significant difference found (p < 0.05)")
else:
    print("  ✗ FAIL TO REJECT H0 — No significant difference")

# PLOT 13 — Gross Profit by Vendor
print("[PLOT 13] Gross Profit by Vendor...")
gp_v = df.groupby('VendorName')['gross_profit'].sum().nlargest(10)
fig, ax = plt.subplots(figsize=(12,6))
c_gp=sns.color_palette('Greens_d',10)
bgp=ax.bar(range(len(gp_v)), gp_v.values, color=c_gp)
ax.set_xticks(range(len(gp_v))); ax.set_xticklabels(gp_v.index, rotation=30, ha='right', fontsize=9)
for bar,val in zip(bgp, gp_v.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+50, fmt(val), ha='center', fontsize=8, fontweight='bold')
ax.set_title('Top 10 Vendors by Gross Profit', fontweight='bold'); ax.set_ylabel('Gross Profit ($)')
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_:f'${x/1e3:.0f}K'))
plt.tight_layout(); plt.savefig('/home/claude/eda_output/plots/13_gross_profit.png',bbox_inches='tight'); plt.close()
print("  ✓ 13_gross_profit.png")

conn.close()
print("\n"+"="*60)
print("✅ COMPLETE EDA FINISHED — 13 Plots + Full SQL + Python Analysis")
print("="*60)
