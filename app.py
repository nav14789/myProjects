import streamlit as st
import sqlite3
import pandas as pd
from datetime import date, timedelta
import math

# ---------- Config ----------
SOAPS = [
    "Almond", "Marigold", "Sandal Turmeric", "Rose",
    "Lavender", "Cucumber", "Aloevera"
]

RAW_MATERIALS_LIST = [
    "Castor Oil", "Shea Butter", "Coconut Oil", "Rice Bran Oil", 
    "Sunflower Oil", "Palmolin Oil", 
    "Almond Fragrance", "Sandalwood Fragrance", "Cucumber Fragrance", 
    "Aloevera Fragrance", "Rose Fragrance", "Lavender Fragrance", "Marigold Fragrance"
]

ADMIN_PASSWORD = "1234"

# ---------- Custom CSS ----------
def load_custom_css():
    st.markdown("""
        <style>
        .stApp { background-color: #f0e4d7; color: #2c3e50; }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] { background-color: #e8dacd; border-right: 1px solid #d9c8b8; }
        [data-testid="stSidebar"] * { color: #2c3e50 !important; }
        [data-testid="stSidebar"] input { background-color: #ffffff !important; color: #2c3e50 !important; border: 1px solid #ccc; }
        
        /* Form inputs styling */
        input, textarea, .stSelectbox div[data-baseweb="select"] { background-color: #ffffff !important; color: #2c3e50 !important; border-radius: 8px; border: 1px solid #ccc; }
        label, .stTextInput label, .stNumberInput label, .stDateInput label, .stSelectbox label { color: #2c3e50 !important; font-weight: 600; }
        
        div[data-baseweb="popover"], div[data-baseweb="menu"] { background-color: #ffffff !important; }
        div[data-baseweb="menu"] li, div[data-baseweb="menu"] div { color: #2c3e50 !important; }
        div[data-baseweb="menu"] li:hover, div[data-baseweb="menu"] div:hover { background-color: #f0f0f0 !important; }
        .stSelectbox div[data-testid="stMarkdownContainer"] p { color: #2c3e50 !important; }

        /* Tabs styling */
        .stTabs [data-baseweb="tab"] { background-color: #fdfbf7 !important; color: #5d4037 !important; border: 1px solid #efebe0; }
        .stTabs [aria-selected="true"] { background-color: #e3f2fd !important; border: 1px solid #90caf9; color: #1565c0 !important; }
        
        /* Buttons styling */
        .stButton > button { background-color: #ffffff !important; color: #2c3e50 !important; border: 1px solid #b0bec5 !important; font-weight: 600; border-radius: 10px; }
        .stButton > button:hover { border-color: #1e88e5 !important; color: #1e88e5 !important; background-color: #f5faff !important; }
        .stButton > button[kind="primary"] { background-color: #d32f2f !important; color: #ffffff !important; border: none !important; }
        .stButton > button[kind="primary"] p { color: #ffffff !important; }
        .stButton > button[kind="primary"]:hover { background-color: #b71c1c !important; color: #ffffff !important; }
        
        h1, h2, h3, h4 { color: #2c3e50 !important; font-family: 'Helvetica Neue', sans-serif; }
        [data-testid="stForm"] { background-color: #ffffff; padding: 25px; border-radius: 12px; border: 1px solid #e0dcd0; }
        [data-testid="stNumberInput"] input { text-align: center; }

        /* --- NEW SUMMARY CARD STYLES (Dark Mode Compatible) --- */
        .summary-card {
            background-color: var(--secondary-background-color);
            border: 1px solid var(--primary-color);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            margin-bottom: 25px;
            text-align: center;
        }
        .summary-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-color);
            margin-bottom: 15px;
            border-bottom: 1px solid var(--primary-color);
            padding-bottom: 10px;
        }
        .stats-container {
            display: flex;
            justify-content: space-around;
            align-items: center;
        }
        .stat-box {
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        .stat-value {
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--primary-color);
        }
        .stat-label {
            font-size: 1rem;
            font-weight: 500;
            color: var(--text-color);
            opacity: 0.8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        /* Color accents that work on light/dark backgrounds */
        .cured-color { color: #2e7d32 !important; } /* Green */
        .queue-color { color: #ef6c00 !important; } /* Orange */
        </style>
    """, unsafe_allow_html=True)

# ---------- DB Helpers ----------
def get_conn():
    conn = sqlite3.connect("pampered_inventory.db", check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS production (id INTEGER PRIMARY KEY AUTOINCREMENT, soap TEXT NOT NULL, prepared_date TEXT NOT NULL, quantity INTEGER NOT NULL CHECK(quantity > 0));""")
    cur.execute("""CREATE TABLE IF NOT EXISTS sales (id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT NOT NULL, platform TEXT NOT NULL CHECK(platform IN ('Amazon','Flipkart','Website')), soap TEXT NOT NULL, quantity INTEGER NOT NULL CHECK(quantity > 0), sale_date TEXT NOT NULL, customer_name TEXT, phone_number TEXT, address TEXT, gift_desc TEXT);""")
    
    cur.execute("PRAGMA table_info(sales)")
    cols = [info[1] for info in cur.fetchall()]
    if "customer_name" not in cols: cur.execute("ALTER TABLE sales ADD COLUMN customer_name TEXT")
    if "phone_number" not in cols: cur.execute("ALTER TABLE sales ADD COLUMN phone_number TEXT")
    if "address" not in cols: cur.execute("ALTER TABLE sales ADD COLUMN address TEXT")
    if "gift_desc" not in cols: cur.execute("ALTER TABLE sales ADD COLUMN gift_desc TEXT")

    cur.execute("""CREATE TABLE IF NOT EXISTS rotten (id INTEGER PRIMARY KEY AUTOINCREMENT, soap TEXT NOT NULL, quantity INTEGER NOT NULL CHECK(quantity > 0), source TEXT NOT NULL CHECK(source IN ('Queue','Cured')), note TEXT, event_date TEXT NOT NULL);""")
    cur.execute("""CREATE TABLE IF NOT EXISTS adjustments (id INTEGER PRIMARY KEY AUTOINCREMENT, soap TEXT NOT NULL, quantity INTEGER NOT NULL, note TEXT, event_date TEXT NOT NULL);""")
    cur.execute("""CREATE TABLE IF NOT EXISTS platform_levels (soap TEXT PRIMARY KEY, amazon INTEGER DEFAULT 0, flipkart INTEGER DEFAULT 0, website INTEGER DEFAULT 0);""")
    cur.execute("""CREATE TABLE IF NOT EXISTS packaging (item_name TEXT PRIMARY KEY, quantity INTEGER DEFAULT 0);""")
    
    # --- NEW TABLES FOR RAW MATERIALS ---
    cur.execute("""CREATE TABLE IF NOT EXISTS raw_materials (item_name TEXT PRIMARY KEY, quantity REAL DEFAULT 0);""")
    cur.execute("""CREATE TABLE IF NOT EXISTS recipes (soap TEXT, ingredient TEXT, amount_needed REAL, batch_size_ref INTEGER, PRIMARY KEY (soap, ingredient));""")

    # Init Packaging
    for soap in SOAPS:
        cur.execute("INSERT OR IGNORE INTO platform_levels (soap, amazon, flipkart, website) VALUES (?, 0, 0, 0)", (soap,))
        cur.execute("INSERT OR IGNORE INTO packaging (item_name, quantity) VALUES (?, 0)", (f"{soap} Card",))
        cur.execute("INSERT OR IGNORE INTO packaging (item_name, quantity) VALUES (?, 0)", (f"{soap} Sticker",))
    cur.execute("INSERT OR IGNORE INTO packaging (item_name, quantity) VALUES ('Gift', 0)")
    cur.execute("UPDATE packaging SET item_name = 'Proud To Be' WHERE item_name = 'Soap Care Guide'")
    cur.execute("INSERT OR IGNORE INTO packaging (item_name, quantity) VALUES ('Proud To Be', 0)")

    # Init Raw Materials
    for item in RAW_MATERIALS_LIST:
        cur.execute("INSERT OR IGNORE INTO raw_materials (item_name, quantity) VALUES (?, 0)", (item,))

    conn.commit()
    conn.close()

# ---------- Write Functions ----------
def get_recipe_for_soap(soap):
    conn = get_conn()
    rows = conn.execute("SELECT ingredient, amount_needed, batch_size_ref FROM recipes WHERE soap = ?", (soap,)).fetchall()
    conn.close()
    return rows

def update_raw_material_stock(item_name, delta):
    conn = get_conn()
    conn.execute("UPDATE raw_materials SET quantity = quantity + ? WHERE item_name = ?", (delta, item_name))
    conn.commit()
    conn.close()

def save_recipe(soap, ingredients_dict, batch_size_ref):
    conn = get_conn()
    conn.execute("DELETE FROM recipes WHERE soap = ?", (soap,))
    for ing, amount in ingredients_dict.items():
        if amount > 0:
            conn.execute("INSERT INTO recipes (soap, ingredient, amount_needed, batch_size_ref) VALUES (?, ?, ?, ?)", 
                         (soap, ing, amount, batch_size_ref))
    conn.commit()
    conn.close()

def insert_production(soap, prepared_date, quantity):
    conn = get_conn()
    try:
        conn.execute("INSERT INTO production (soap, prepared_date, quantity) VALUES (?, ?, ?)", (soap, prepared_date.isoformat(), int(quantity)))
        
        recipe_rows = conn.execute("SELECT ingredient, amount_needed, batch_size_ref FROM recipes WHERE soap = ?", (soap,)).fetchall()
        if recipe_rows:
            for row in recipe_rows:
                ingredient = row["ingredient"]
                amount_per_ref = row["amount_needed"]
                ref_size = row["batch_size_ref"]
                deduction = (amount_per_ref / ref_size) * int(quantity)
                conn.execute("UPDATE raw_materials SET quantity = quantity - ? WHERE item_name = ?", (deduction, ingredient))
        conn.commit()
        return True
    finally:
        conn.close()

def delete_production_batches(ids_to_delete):
    if not ids_to_delete: return
    conn = get_conn()
    try:
        placeholders = ','.join('?' for _ in ids_to_delete)
        rows_to_del = conn.execute(f"SELECT soap, quantity FROM production WHERE id IN ({placeholders})", ids_to_delete).fetchall()
        
        for prod_row in rows_to_del:
            soap_name = prod_row["soap"]
            qty_produced = prod_row["quantity"]
            recipe_rows = conn.execute("SELECT ingredient, amount_needed, batch_size_ref FROM recipes WHERE soap = ?", (soap_name,)).fetchall()
            for r_row in recipe_rows:
                ingredient = r_row["ingredient"]
                amount_per_ref = r_row["amount_needed"]
                ref_size = r_row["batch_size_ref"]
                add_back = (amount_per_ref / ref_size) * qty_produced
                conn.execute("UPDATE raw_materials SET quantity = quantity + ? WHERE item_name = ?", (add_back, ingredient))
        
        conn.execute(f"DELETE FROM production WHERE id IN ({placeholders})", ids_to_delete)
        conn.commit()
    finally:
        conn.close()

def insert_sale(order_id, platform, soap, quantity, sale_date, c_name, c_phone, c_addr, gift_desc):
    conn = get_conn()
    quantity = int(quantity)
    order_id = order_id.strip()
    try:
        cursor = conn.execute("SELECT COUNT(*) FROM sales WHERE order_id = ?", (order_id,))
        is_new_order = (cursor.fetchone()[0] == 0)
        conn.execute("INSERT INTO sales (order_id, platform, soap, quantity, sale_date, customer_name, phone_number, address, gift_desc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", 
                     (order_id, platform, soap, quantity, sale_date.isoformat(), c_name, c_phone, c_addr, gift_desc))
        col_map = {"Amazon": "amazon", "Flipkart": "flipkart", "Website": "website"}
        if platform in col_map:
            col = col_map[platform]
            conn.execute(f"UPDATE platform_levels SET {col} = {col} - ? WHERE soap = ?", (quantity, soap))
        conn.execute("UPDATE packaging SET quantity = quantity - ? WHERE item_name = 'Proud To Be'", (quantity,))
        conn.execute("UPDATE packaging SET quantity = quantity - ? WHERE item_name = ?", (quantity, f"{soap} Card"))
        conn.execute("UPDATE packaging SET quantity = quantity - ? WHERE item_name = ?", (quantity, f"{soap} Sticker"))
        if is_new_order:
            conn.execute("UPDATE packaging SET quantity = quantity - 1 WHERE item_name = 'Gift'")
        conn.commit()
        return is_new_order
    finally:
        conn.close()

def insert_rotten(soap, quantity, source, note, event_date):
    conn = get_conn()
    conn.execute("INSERT INTO rotten (soap, quantity, source, note, event_date) VALUES (?, ?, ?, ?, ?)", (soap, int(quantity), source, note.strip() if note else None, event_date.isoformat()))
    conn.commit()
    conn.close()

def insert_adjustment(soap, quantity, note):
    conn = get_conn()
    conn.execute("INSERT INTO adjustments (soap, quantity, note, event_date) VALUES (?, ?, ?, ?)", (soap, int(quantity), note, date.today().isoformat()))
    conn.commit()
    conn.close()

def update_packaging_stock(updates):
    conn = get_conn()
    for item, qty in updates.items(): conn.execute("UPDATE packaging SET quantity = ? WHERE item_name = ?", (qty, item))
    conn.commit()
    conn.close()

def update_raw_materials_manual(updates):
    conn = get_conn()
    for item, qty in updates.items(): conn.execute("UPDATE raw_materials SET quantity = ? WHERE item_name = ?", (qty, item))
    conn.commit()
    conn.close()

def update_platform_levels(edited_df):
    conn = get_conn()
    for index, row in edited_df.iterrows():
        conn.execute("UPDATE platform_levels SET amazon = ?, flipkart = ?, website = ? WHERE soap = ?", (row['Amazon Stock'], row['Flipkart Stock'], row['Website Stock'], row['Soap']))
    conn.commit()
    conn.close()

def revert_entire_order(order_id):
    conn = get_conn()
    try:
        rows = conn.execute("SELECT * FROM sales WHERE order_id = ?", (order_id,)).fetchall()
        if not rows: return False
        for row in rows:
            qty = row['quantity']
            soap = row['soap']
            platform = row['platform']
            col_map = {"Amazon": "amazon", "Flipkart": "flipkart", "Website": "website"}
            if platform in col_map:
                col = col_map[platform]
                conn.execute(f"UPDATE platform_levels SET {col} = {col} + ? WHERE soap = ?", (qty, soap))
            conn.execute("UPDATE packaging SET quantity = quantity + ? WHERE item_name = 'Proud To Be'", (qty,))
            conn.execute("UPDATE packaging SET quantity = quantity + ? WHERE item_name = ?", (qty, f"{soap} Card"))
            conn.execute("UPDATE packaging SET quantity = quantity + ? WHERE item_name = ?", (qty, f"{soap} Sticker"))
        conn.execute("UPDATE packaging SET quantity = quantity + 1 WHERE item_name = 'Gift'")
        conn.execute("DELETE FROM sales WHERE order_id = ?", (order_id,))
        conn.commit()
        return True
    finally:
        conn.close()

# ---------- CALLBACKS ----------
def record_sale_callback():
    oid = st.session_state.get("sale_oid", "").strip()
    c_name = st.session_state.get("sale_c_name", "")
    c_phone = st.session_state.get("sale_c_phone", "")
    c_addr = st.session_state.get("sale_c_addr", "")
    gift_desc = st.session_state.get("sale_gift_desc", "")
    plt = st.session_state.get("sale_platform", "Amazon")
    sd = st.session_state.get("sale_date", date.today())
    
    cart_counts = {}
    for soap in SOAPS:
        key = f"cart_{soap}"
        qty = st.session_state.get(key, 0)
        if qty > 0: cart_counts[soap] = qty

    if not cart_counts:
        st.session_state["sale_error"] = "⚠️ Please add at least 1 soap to the cart."
        return
    if not oid:
        st.session_state["sale_error"] = "⚠️ Order ID is required."
        return

    try:
        for soap_name, quantity in cart_counts.items():
            insert_sale(oid, plt, soap_name, quantity, sd, c_name, c_phone, c_addr, gift_desc)
        
        st.session_state["show_success"] = True
        st.session_state["success_oid"] = oid
        st.session_state["success_count"] = sum(cart_counts.values())
        st.session_state["sale_error"] = None

        st.session_state["sale_oid"] = ""
        st.session_state["sale_c_name"] = ""
        st.session_state["sale_c_phone"] = ""
        st.session_state["sale_c_addr"] = ""
        st.session_state["sale_gift_desc"] = ""
        st.session_state["sale_date"] = date.today()
        for soap in SOAPS:
            st.session_state[f"cart_{soap}"] = 0
            
    except Exception as e:
        st.session_state["sale_error"] = f"Error saving: {str(e)}"

# ---------- Read & Compute ----------
def fetch_df(query, params=()):
    conn = get_conn()
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def get_packaging(): return fetch_df("SELECT item_name, quantity FROM packaging")
def get_raw_materials(): return fetch_df("SELECT item_name, quantity FROM raw_materials ORDER BY item_name")
def get_production():
    df = fetch_df("SELECT id, soap, prepared_date, quantity FROM production ORDER BY prepared_date DESC;")
    if not df.empty and "prepared_date" in df.columns: df["prepared_date"] = pd.to_datetime(df["prepared_date"]).dt.date
    return df
def get_sales():
    df = fetch_df("SELECT id, order_id, platform, soap, quantity, sale_date, customer_name, phone_number, address, gift_desc FROM sales ORDER BY sale_date DESC;")
    if not df.empty and "sale_date" in df.columns: df["sale_date"] = pd.to_datetime(df["sale_date"]).dt.date
    return df
def get_rotten(): return fetch_df("SELECT id, soap, quantity, source, note, event_date FROM rotten ORDER BY event_date DESC;")
def get_adjustments(): return fetch_df("SELECT soap, quantity FROM adjustments")
def get_platform_levels(): return fetch_df("SELECT soap, amazon, flipkart, website FROM platform_levels")

def compute_inventory(curing_days):
    today = date.today()
    prod = get_production(); sales = fetch_df("SELECT * FROM sales"); rot = get_rotten(); adjs = get_adjustments(); platforms = get_platform_levels()
    cured_counts = {s: 0 for s in SOAPS}
    queue_rows = [] 

    if not prod.empty:
        prod["cure_date"] = prod["prepared_date"].apply(lambda d: d + timedelta(days=curing_days))
        for _, row in prod.iterrows():
            days_left = (row["cure_date"] - today).days
            if days_left <= 0: cured_counts[row["soap"]] += int(row["quantity"])
            else:
                queue_rows.append({"id": row["id"], "Select": False, "Soap": row["soap"], "Quantity": int(row["quantity"]), "Prepared": row["prepared_date"], "Cure Date": row["cure_date"], "Days Left": days_left})

    if not rot.empty:
        for _, row in rot.iterrows():
            if row["source"] == "Cured": cured_counts[row["soap"]] -= int(row["quantity"])
    if not sales.empty:
        for _, row in sales.iterrows(): cured_counts[row["soap"]] -= int(row["quantity"])
    if not adjs.empty:
        for _, row in adjs.iterrows():
            if row["soap"] in cured_counts: cured_counts[row["soap"]] += int(row["quantity"])

    cured_final = {k: v for k, v in cured_counts.items()} 
    cured_df = pd.DataFrame({"Soap": list(cured_final.keys()), "Total Cured Inventory": list(cured_final.values())})

    if not platforms.empty:
        cured_df = cured_df.merge(platforms, left_on="Soap", right_on="soap", how="left").drop(columns=["soap"]).fillna(0)
    else:
        cured_df["amazon"] = 0; cured_df["flipkart"] = 0; cured_df["website"] = 0
    
    cured_df = cured_df.rename(columns={"amazon": "Amazon Stock", "flipkart": "Flipkart Stock", "website": "Website Stock"})
    cured_df = cured_df[["Soap", "Total Cured Inventory", "Amazon Stock", "Flipkart Stock", "Website Stock"]].sort_values("Soap")

    if queue_rows: queue_df = pd.DataFrame(queue_rows).sort_values(by=["Days Left", "id"])
    else: queue_df = pd.DataFrame(columns=["id", "Select", "Soap", "Quantity", "Prepared", "Cure Date", "Days Left"])
    return cured_df, queue_df

def get_stock_projection(target_date, curing_days):
    prod = get_production(); sales = fetch_df("SELECT * FROM sales"); rot = get_rotten(); adjs = get_adjustments()
    counts = {s: 0 for s in SOAPS}
    if not prod.empty:
        prod["cure_date"] = prod["prepared_date"].apply(lambda d: d + timedelta(days=curing_days))
        ready = prod[prod["cure_date"] <= target_date]
        for _, row in ready.iterrows(): counts[row["soap"]] += int(row["quantity"])
    if not sales.empty:
        for _, row in sales.iterrows(): counts[row["soap"]] -= int(row["quantity"])
    if not rot.empty:
        for _, row in rot.iterrows(): counts[row["soap"]] -= int(row["quantity"])
    if not adjs.empty:
        for _, row in adjs.iterrows():
            if row["soap"] in counts: counts[row["soap"]] += int(row["quantity"])
    return {k: max(0, v) for k, v in counts.items()}

def get_sales_pivoted():
    df = fetch_df("SELECT * FROM sales")
    if df.empty: return pd.DataFrame()
    if "sale_date" in df.columns: df["sale_date"] = pd.to_datetime(df["sale_date"]).dt.date
    pivot_cols = ['order_id', 'platform', 'sale_date', 'customer_name', 'phone_number', 'address', 'gift_desc']
    df_pivot = df.pivot_table(index=pivot_cols, columns='soap', values='quantity', fill_value=0, aggfunc='sum').reset_index()
    for soap in SOAPS:
        if soap not in df_pivot.columns: df_pivot[soap] = 0
    df_pivot = df_pivot.sort_values(by="sale_date", ascending=False)
    df_pivot.insert(0, "Select", False)
    return df_pivot

# ---------- PREDICTION LOGIC ----------
def get_prediction_advice(curing_days):
    sales_df = fetch_df("SELECT soap, quantity, sale_date FROM sales")
    if sales_df.empty: return [] 
    
    sales_df["sale_date"] = pd.to_datetime(sales_df["sale_date"]).dt.date
    cutoff_date = date.today() - timedelta(days=20)
    recent_sales = sales_df[sales_df["sale_date"] >= cutoff_date]
    
    advice_list = []
    c_df, q_df = compute_inventory(curing_days)
    
    for soap in SOAPS:
        total_sold_20d = recent_sales[recent_sales["soap"] == soap]["quantity"].sum()
        daily_rate = total_sold_20d / 20.0
        
        if daily_rate > 0:
            cured_now = 0
            c_row = c_df[c_df["Soap"] == soap]
            if not c_row.empty: cured_now = c_row.iloc[0]["Total Cured Inventory"]
            
            queue_now = 0
            if not q_df.empty:
                queue_now = q_df[q_df["Soap"] == soap]["Quantity"].sum()
            
            total_pipeline = cured_now + queue_now
            needed_stock = daily_rate * curing_days
            
            # --- GAP LOGIC ---
            gap_days = 0
            gap_message = ""
            if cured_now < needed_stock:
                days_until_empty = cured_now / daily_rate if daily_rate > 0 else 0
                gap_days = int(curing_days - days_until_empty)
                if gap_days > 0 and queue_now == 0:
                    gap_message = f"📉 **Unavoidable Gap:** You will be out of stock for approx **{gap_days} days** before today's batch cures."
            
            if total_pipeline < needed_stock:
                shortfall = math.ceil(needed_stock - total_pipeline)
                advice_list.append({
                    "Soap": soap,
                    "Advice": f"Prepare {shortfall} bars TODAY",
                    "GapMsg": gap_message,
                    "Details": {
                        "Daily Sales": round(daily_rate, 2),
                        "Curing Days": curing_days,
                        "Need": round(needed_stock, 1),
                        "Have (Cured+Queue)": total_pipeline
                    }
                })
    return advice_list

def make_pretty(df):
    return df.style.set_properties(**{'background-color': '#ffffff', 'color': '#2c3e50', 'border-color': '#e1e5eb'}).set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#e3f2fd'), ('color', '#1565c0'), ('font-weight', 'bold'), ('border-bottom', '2px solid #90caf9')]},
        {'selector': 'tr:hover', 'props': [('background-color', '#fff9e6')]}
    ])

def style_low_inventory_pretty(df, col, threshold):
    def highlight_low(s): return ['color: #d32f2f; font-weight: bold; background-color: #ffebee;' if v < threshold else '' for v in s]
    styler = make_pretty(df)
    return styler.apply(highlight_low, subset=[col])

def convert_df_to_csv(df): return df.to_csv(index=False).encode('utf-8')

# ---------- DIALOGS ----------
@st.dialog("⚠️ Low Inventory Report")
def show_low_inventory_popup(curing_days, threshold):
    cured_df, _ = compute_inventory(curing_days)
    low_soaps = cured_df[cured_df["Total Cured Inventory"] < threshold][["Soap", "Total Cured Inventory"]]
    
    pack_df = get_packaging()
    low_pack = pack_df[pack_df["quantity"] < threshold]
    
    raw_df = get_raw_materials()
    low_raw = raw_df[raw_df["quantity"] < (threshold * 10)] 
    
    if low_soaps.empty and low_pack.empty and low_raw.empty:
        st.success(f"✅ All items are healthy!")
    else:
        if not low_soaps.empty:
            st.markdown(f"### 🧼 Soaps (< {threshold})")
            st.dataframe(low_soaps, hide_index=True, use_container_width=True)
        if not low_pack.empty:
            st.markdown(f"### 📦 Packaging (< {threshold})")
            st.dataframe(low_pack, hide_index=True, use_container_width=True)
        if not low_raw.empty:
            st.markdown(f"### 🛢️ Low Raw Materials (< {threshold*10}g)")
            st.dataframe(low_raw, hide_index=True, use_container_width=True)

@st.dialog("🔮 Prediction to Prepare")
def show_prediction_popup(curing_days):
    st.markdown(f"**Analysis:** Last 20 days sales vs {curing_days}-day curing cycle.")
    advice_list = get_prediction_advice(curing_days)
    
    if not advice_list:
        st.success("🎉 Stock pipeline is healthy! No immediate production needed.")
    else:
        for item in advice_list:
            with st.container(border=True):
                st.markdown(f"### ⚠️ **{item['Soap']}**: {item['Advice']}")
                if item["GapMsg"]:
                    st.error(item["GapMsg"])
                with st.expander("ℹ️ See Calculation Logic"):
                    d = item['Details']
                    st.write(f"**Avg Daily Sales:** {d['Daily Sales']}")
                    st.write(f"**Required Safety Stock:** {d['Need']}")
                    st.write(f"**Current Pipeline:** {d['Have (Cured+Queue)']}")
                    st.divider()
                    st.markdown(f"**Action:** {d['Need']} - {d['Have (Cured+Queue)']} = **{item['Advice']}**")

@st.dialog("✅ Order Recorded Successfully!")
def show_success_modal(oid, total_items):
    st.markdown(f"### Order ID: **{oid}**")
    st.markdown(f"**{total_items}** items deducted from inventory.")
    st.success("Form has been reset for the next order.")
    if st.button("OK", type="primary"):
        st.rerun()

# ---------- UI ----------
def main():
    st.set_page_config(page_title="Pampered Organically", page_icon="🧼", layout="wide")
    load_custom_css()
    init_db()

    with st.sidebar:
        st.header("Navigation")
        dashboard_mode = st.radio("Go to:", ["🧼 Soap Inventory", "🛢️ Raw Materials", "⚗️ Recipe Config", "🎁 Packaging & Gifts"], index=0)
        st.divider()
        st.header("Settings")
        curing_days = st.number_input("Curing Days", value=30)
        low_threshold = st.number_input("Low Stock Alert (Units)", 20)
        
        rows_per_page = st.number_input("Rows Per Page", min_value=5, value=25)
        
        st.divider()
        edit_mode = st.toggle("Enable Data Editing")
        is_admin = False
        if edit_mode:
            pwd = st.text_input("Admin Password", type="password")
            if pwd == ADMIN_PASSWORD:
                is_admin = True
                st.success("Unlocked!")
            elif pwd != "":
                st.error("Wrong password.")
        st.divider()
        if st.button("🔄 Refresh Data"): st.rerun()

    col_title, col_btn = st.columns([4, 2])
    with col_title: st.title("🌿 Pampered Organically")
    with col_btn:
        st.write("") 
        st.write("") 
        b1, b2 = st.columns(2)
        cured_df, _ = compute_inventory(curing_days)
        low_soaps_count = len(cured_df[cured_df["Total Cured Inventory"] < low_threshold])
        pack_df = get_packaging()
        low_pack_count = len(pack_df[pack_df["quantity"] < low_threshold])
        raw_df = get_raw_materials()
        low_raw_count = len(raw_df[raw_df["quantity"] < (low_threshold * 10)])
        
        total_low = low_soaps_count + low_pack_count + low_raw_count
        
        with b1:
            if total_low > 0:
                if st.button(f"⚠️ Low Stock ({total_low})", type="primary"): show_low_inventory_popup(curing_days, low_threshold)
            else:
                if st.button("✅ Stock Healthy", type="secondary"): show_low_inventory_popup(curing_days, low_threshold)
        
        with b2:
            if st.button("🔮 Prediction"): show_prediction_popup(curing_days)

    st.markdown("### Inventory & Order Management System")

    # --- SUCCESS MODAL TRIGGER ---
    if "show_success" in st.session_state and st.session_state["show_success"]:
        show_success_modal(st.session_state["success_oid"], st.session_state["success_count"])
        st.session_state["show_success"] = False
        st.session_state["success_oid"] = ""
        st.session_state["success_count"] = 0

    # --- ERROR TRIGGER ---
    if "sale_error" in st.session_state and st.session_state["sale_error"]:
        st.error(st.session_state["sale_error"])
        st.session_state["sale_error"] = None 

    if dashboard_mode == "🧼 Soap Inventory":
        tabs = st.tabs(["📊 Dashboard", "📜 Orders", "➕ Production", "💰 Record Sale", "🗑️ Mark Damaged"])

        with tabs[0]:
            cured_df, queue_df = compute_inventory(curing_days)

            # --- SELECTION & CARD LOGIC ---
            # 1. Determine Selection (Only works in View mode, not Edit mode)
            selected_soap_name = None
            
            # If we are in "View Mode" (not admin/edit), we can use the selection from the dataframe
            # We access the session state for the dataframe selection if it exists
            if not is_admin and "inventory_table" in st.session_state:
                selection = st.session_state["inventory_table"].get("selection", {})
                if selection and "rows" in selection and selection["rows"]:
                    idx = selection["rows"][0]
                    selected_soap_name = cured_df.iloc[idx]["Soap"]

            # 2. Calculate Totals based on selection
            if selected_soap_name:
                card_title = f"🧼 {selected_soap_name}"
                # Get specific cured count
                c_row = cured_df[cured_df["Soap"] == selected_soap_name]
                cured_display = c_row.iloc[0]["Total Cured Inventory"] if not c_row.empty else 0
                # Get specific queue count
                queue_display = queue_df[queue_df["Soap"] == selected_soap_name]["Quantity"].sum()
            else:
                card_title = "🏭 All Soaps Overview"
                cured_display = cured_df["Total Cured Inventory"].sum()
                queue_display = queue_df["Quantity"].sum()

            # 3. Render Beautiful Card
            st.markdown(f"""
                <div class="summary-card">
                    <div class="summary-header">{card_title}</div>
                    <div class="stats-container">
                        <div class="stat-box">
                            <span class="stat-value cured-color">{cured_display}</span>
                            <span class="stat-label">Ready (Cured)</span>
                        </div>
                        <div class="stat-box">
                            <span class="stat-value queue-color">{queue_display}</span>
                            <span class="stat-label">In Queue</span>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            # -------------------------------

            colA, colB = st.columns([1.5, 1])
            with colA:
                st.markdown("#### Inventory Levels")
                if is_admin:
                    st.info("💡 Edits to 'Total' create adjustments.")
                    edited_cured = st.data_editor(
                        cured_df, 
                        key="cured_editor", 
                        use_container_width=True, 
                        disabled=["Soap"], 
                        hide_index=True,
                        column_config={
                            "Total Cured Inventory": st.column_config.NumberColumn("Total Cured", help="Physical Stock"),
                        }
                    )
                    if st.button("Save Stock Updates", type="primary"):
                        update_platform_levels(edited_cured)
                        for i, row in edited_cured.iterrows():
                            soap_name = row["Soap"]
                            new_qty = row["Total Cured Inventory"]
                            original_row = cured_df[cured_df["Soap"] == soap_name]
                            if not original_row.empty:
                                diff = new_qty - original_row.iloc[0]["Total Cured Inventory"]
                                if diff != 0: insert_adjustment(soap_name, diff, "Manual Dashboard Update")
                        st.success("Stocks updated!")
                        st.rerun()
                else:
                    # VIEW MODE: Enable Selection
                    st.caption("👆 Click a row to see details in the card above.")
                    styler = style_low_inventory_pretty(cured_df, "Total Cured Inventory", low_threshold)
                    
                    # We pass the styler, but we also enable selection
                    st.dataframe(
                        styler, 
                        use_container_width=True, 
                        hide_index=True,
                        on_select="rerun",  # Triggers rerun to update the card
                        selection_mode="single-row",
                        key="inventory_table"
                    )
                
                st.download_button("📥 Download Stock", convert_df_to_csv(cured_df), "stock.csv", "text/csv", type="primary")

            with colB:
                st.markdown("#### In Curing Queue")
                
                # --- PAGINATION LOGIC ---
                if "queue_page" not in st.session_state: st.session_state.queue_page = 1

                # Apply Filter to Queue if a soap is selected
                if selected_soap_name:
                    display_queue = queue_df[queue_df["Soap"] == selected_soap_name]
                    st.info(f"Showing queue for: **{selected_soap_name}**")
                else:
                    display_queue = queue_df

                total_rows = len(display_queue)
                total_pages = math.ceil(total_rows / rows_per_page)
                if total_pages < 1: total_pages = 1
                if st.session_state.queue_page > total_pages: st.session_state.queue_page = total_pages
                
                start_idx = (st.session_state.queue_page - 1) * rows_per_page
                end_idx = start_idx + rows_per_page
                paginated_queue = display_queue.iloc[start_idx:end_idx]

                if is_admin:
                    st.warning("🗑️ **Delete:** Restores ingredients to Raw Materials.")
                    edited_queue = st.data_editor(
                        paginated_queue, 
                        key="queue_edit_box", 
                        use_container_width=True, 
                        hide_index=True, 
                        column_config={"id": None, "Select": st.column_config.CheckboxColumn(required=True)}
                    )
                    if st.button("🗑️ Delete Selected Batches", type="primary"):
                        rows_to_delete = edited_queue[edited_queue["Select"] == True]
                        if not rows_to_delete.empty:
                            ids_to_del = rows_to_delete["id"].tolist()
                            delete_production_batches(ids_to_del)
                            st.success(f"Deleted {len(ids_to_del)} batches. Raw Materials Restored.")
                            st.rerun()
                        else: st.info("Please select rows to delete first.")
                else:
                    try:
                        styler = make_pretty(paginated_queue.drop(columns=["id", "Select"])).background_gradient(cmap="Reds", subset=["Days Left"])
                        st.dataframe(styler, use_container_width=True, hide_index=True)
                    except:
                        st.dataframe(make_pretty(paginated_queue.drop(columns=["id", "Select"])), use_container_width=True, hide_index=True)
                
                # --- PAGINATION BUTTONS ---
                if total_pages > 1:
                    c_prev, c_page, c_next = st.columns([1, 2, 1])
                    with c_prev:
                        if st.button("Previous", disabled=(st.session_state.queue_page == 1)):
                            st.session_state.queue_page -= 1
                            st.rerun()
                    with c_page:
                        st.markdown(f"<div style='text-align: center; padding-top: 5px;'>Page <b>{st.session_state.queue_page}</b> of <b>{total_pages}</b></div>", unsafe_allow_html=True)
                    with c_next:
                        if st.button("Next", disabled=(st.session_state.queue_page == total_pages)):
                            st.session_state.queue_page += 1
                            st.rerun()

                st.download_button("📥 Download Queue", convert_df_to_csv(queue_df.drop(columns=["id", "Select"])), "queue.csv", "text/csv", type="primary")

            st.markdown("---")
            st.subheader("🔮 Inventory Forecasting")
            fc1, fc2 = st.columns(2)
            with fc1:
                st.markdown("**Check Specific Date**")
                target_date = st.date_input("Select Future Date", value=date.today() + timedelta(days=7))
                if st.button("Check Availability"):
                    forecast = get_stock_projection(target_date, curing_days)
                    f_df = pd.DataFrame(list(forecast.items()), columns=["Soap", "Projected Stock"])
                    st.dataframe(make_pretty(f_df), use_container_width=True, hide_index=True)
            with fc2:
                st.markdown("**3-Day Summary Outlook**")
                d1 = date.today() + timedelta(days=1); d2 = date.today() + timedelta(days=2); d3 = date.today() + timedelta(days=3)
                s1 = get_stock_projection(d1, curing_days); s2 = get_stock_projection(d2, curing_days); s3 = get_stock_projection(d3, curing_days)
                summary_data = []
                for soap in SOAPS:
                    summary_data.append({"Soap": soap, f"{d1.strftime('%b %d')}": s1.get(soap, 0), f"{d2.strftime('%b %d')}": s2.get(soap, 0), f"{d3.strftime('%b %d')}": s3.get(soap, 0)})
                st.dataframe(make_pretty(pd.DataFrame(summary_data)), use_container_width=True, hide_index=True)

        with tabs[1]:
            st.markdown("#### Order History")
            sales_pivot = get_sales_pivoted()
            search_query = st.text_input("🔍 Search Orders", placeholder="Type Order ID, Customer Name, Platform, etc...", key="search_box")
            
            if not sales_pivot.empty:
                if search_query:
                    mask = sales_pivot.astype(str).apply(lambda x: x.str.contains(search_query, case=False)).any(axis=1)
                    sales_display_data = sales_pivot[mask]
                else:
                    sales_display_data = sales_pivot

                default_cols = ['order_id', 'sale_date', 'platform', 'gift_desc'] + SOAPS
                all_cols = [c for c in sales_pivot.columns if c not in ['Select']]
                selected_cols = st.multiselect("Show/Hide Columns", options=all_cols, default=default_cols)
                final_cols = ['Select'] + selected_cols
                
                if is_admin:
                    st.warning("🗑️ **Delete:** Check box. Deletes ENTIRE order & restores stock.")
                    edited_sales = st.data_editor(
                        sales_display_data[final_cols], 
                        key="sales_edit", 
                        num_rows="fixed",
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Select": st.column_config.CheckboxColumn(required=True),
                            "sale_date": st.column_config.DateColumn("Date"),
                            "gift_desc": "Gift",
                            "order_id": "Order ID"
                        }
                    )
                    if st.button("🗑️ Delete Selected Orders", type="primary"):
                        rows_to_delete = edited_sales[edited_sales["Select"] == True]
                        if not rows_to_delete.empty:
                            for _, row in rows_to_delete.iterrows():
                                revert_entire_order(row["order_id"])
                            st.success("Orders Deleted & Stock Reverted.")
                            st.rerun()
                else:
                    display_cols = [c for c in final_cols if c != 'Select']
                    st.dataframe(make_pretty(sales_display_data[display_cols]), use_container_width=True, hide_index=True)
                st.download_button("📥 Download Orders", convert_df_to_csv(sales_display_data.drop(columns=["Select"], errors='ignore')), "orders.csv", "text/csv", type="primary")
            else:
                st.info("No orders found.")

        with tabs[2]:
            st.markdown("#### Add Batch")
            st.info("💡 Ingredients will be automatically deducted based on Recipe Config.")
            with st.form("p_form"):
                c1,c2,c3 = st.columns(3)
                s = c1.selectbox("Soap", SOAPS)
                d = c2.date_input("Date", date.today())
                q = c3.number_input("Qty Produced", 1)
                if st.form_submit_button("Save", type="primary"):
                    insert_production(s, d, q)
                    st.success("Saved & Raw Materials Deducted!")
                    st.rerun()
            st.divider()
            prod_df = get_production()
            if is_admin:
                st.info("✏️ Edit logs directly.")
                edited_prod = st.data_editor(prod_df, key="prod_edit", num_rows="dynamic", use_container_width=True, column_config={"prepared_date": st.column_config.DateColumn("Date"), "soap": st.column_config.SelectboxColumn("Soap", options=SOAPS)})
            else:
                st.dataframe(make_pretty(prod_df), use_container_width=True, hide_index=True)

        with tabs[3]:
            st.markdown("#### Quick Sale Entry")
            st.caption("Deductions: 1 'Proud To Be', 1 Card, 1 Sticker (per soap) & 1 Gift (per ORDER).")
            
            with st.form("sale_form"):
                st.markdown("**Order Info**")
                c1, c2 = st.columns(2)
                # KEYS ARE CRITICAL FOR RESETTING
                oid = c1.text_input("Order ID", placeholder="e.g. 101", key="sale_oid")
                plt = c1.selectbox("Platform", ["Amazon", "Flipkart", "Website"], key="sale_platform")
                sd = c2.date_input("Sale Date", date.today(), key="sale_date")
                
                st.markdown("**Shopping Cart**")
                cols = st.columns(4)
                cart_counts = {}
                for i, soap in enumerate(SOAPS):
                    with cols[i % 4]:
                        qty = st.number_input(f"{soap}", min_value=0, value=0, key=f"cart_{soap}")
                        if qty > 0: cart_counts[soap] = qty
                
                st.markdown("---")
                st.markdown("**Customer Details (Optional)**")
                cust_1, cust_2 = st.columns(2)
                c_name = cust_1.text_input("Customer Name", key="sale_c_name")
                c_phone = cust_2.text_input("Phone Number", key="sale_c_phone")
                c_addr = st.text_area("Shipping Address", height=80, key="sale_c_addr")
                gift_desc = st.text_input("Gift Description", placeholder="e.g. Mini Rose Soap", key="sale_gift_desc")
                
                st.form_submit_button("Record Sale", type="primary", on_click=record_sale_callback)

        with tabs[4]:
            st.markdown("#### Mark Damaged")
            with st.form("rot_form"):
                c1,c2,c3 = st.columns([2,2,1])
                sr = c1.selectbox("Soap", SOAPS)
                src = c2.selectbox("Source", ["Queue", "Cured"])
                rq = c3.number_input("Qty", 1)
                note = st.text_area("Reason")
                if st.form_submit_button("Deduct", type="primary"):
                    insert_rotten(sr, rq, src, note, date.today())
                    st.success("Deducted")
                    st.rerun()

    # --- RAW MATERIALS TAB ---
    elif dashboard_mode == "🛢️ Raw Materials":
        st.title("🛢️ Raw Material Inventory")
        st.markdown("**Units are in Grams (g)**")
        
        raw_df = get_raw_materials()
        
        # Split into Oils and Fragrances for better view
        oils = raw_df[raw_df["item_name"].str.contains("Oil|Butter")]
        fragrances = raw_df[raw_df["item_name"].str.contains("Fragrance")]
        
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Base Oils & Butters")
            st.dataframe(make_pretty(oils), use_container_width=True, hide_index=True)
        with c2:
            st.subheader("Fragrances")
            st.dataframe(make_pretty(fragrances), use_container_width=True, hide_index=True)
            
        st.divider()
        st.subheader("📝 Update Stock")
        with st.form("raw_update"):
            c_i, c_q = st.columns(2)
            item_sel = c_i.selectbox("Ingredient", RAW_MATERIALS_LIST)
            qty_input = c_q.number_input("New Total Quantity (g)", min_value=0.0, step=100.0)
            if st.form_submit_button("Update Stock", type="primary"):
                update_raw_materials_manual({item_sel: qty_input})
                st.success(f"Updated {item_sel} to {qty_input}g")
                st.rerun()

    # --- RECIPE CONFIGURATION TAB ---
    elif dashboard_mode == "⚗️ Recipe Config":
        st.title("⚗️ Recipe Configuration")
        st.info("Configure how much oil/fragrance is needed for a standard batch. This logic is used to deduct stock when you add production.")
        
        selected_soap_recipe = st.selectbox("Select Soap to Configure", SOAPS)
        
        # Load existing recipe if any
        existing_recipe = get_recipe_for_soap(selected_soap_recipe)
        current_map = {row['ingredient']: row['amount_needed'] for row in existing_recipe}
        current_ref_size = existing_recipe[0]['batch_size_ref'] if existing_recipe else 10 # Default to 10 if new
        
        with st.form("recipe_form"):
            ref_size_input = st.number_input(f"Standard Batch Size (e.g. 10 bars of {selected_soap_recipe})", value=current_ref_size, min_value=1)
            st.markdown(f"**Ingredients required for {ref_size_input} bars:**")
            
            inputs = {}
            # Display form for all ingredients
            # Use columns to make it compact
            r_cols = st.columns(3)
            for i, ingredient in enumerate(RAW_MATERIALS_LIST):
                with r_cols[i % 3]:
                    val = current_map.get(ingredient, 0.0)
                    inputs[ingredient] = st.number_input(f"{ingredient} (g)", value=float(val), step=10.0, key=f"rec_{ingredient}")
            
            if st.form_submit_button("Save Recipe", type="primary"):
                save_recipe(selected_soap_recipe, inputs, ref_size_input)
                st.success(f"Recipe for {selected_soap_recipe} saved!")
                st.rerun()

    elif dashboard_mode == "🎁 Packaging & Gifts":
        st.title("🎁 Packaging Inventory")
        pack_df = get_packaging()
        common_df = pack_df[pack_df["item_name"].isin(["Gift", "Proud To Be"])]
        specific_df = pack_df[~pack_df["item_name"].isin(["Gift", "Proud To Be"])].sort_values("item_name")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📦 General Items")
            if not common_df.empty:
                for _, row in common_df.iterrows(): st.metric(label=row["item_name"], value=row["quantity"])
            st.divider()
            with st.form("pack_common"):
                st.markdown("**Update Stock**")
                item_c = st.selectbox("Item", ["Gift", "Proud To Be"])
                qty_c = st.number_input("New Total Quantity", min_value=0)
                if st.form_submit_button("Update General Stock", type="primary"):
                    update_packaging_stock({item_c: qty_c})
                    st.success("Updated!")
                    st.rerun()
        with col2:
            st.markdown("### 🏷️ Soap Specific Items")
            st.dataframe(make_pretty(specific_df), use_container_width=True, height=400, hide_index=True)
            st.divider()
            with st.form("pack_specific"):
                st.markdown("**Update Stock**")
                c_s, c_t = st.columns(2)
                soap_pk = c_s.selectbox("Soap", SOAPS)
                type_pk = c_t.selectbox("Type", ["Card", "Sticker"])
                qty_pk = st.number_input("New Total Quantity", min_value=0)
                if st.form_submit_button("Update Specific Stock", type="primary"):
                    full_name = f"{soap_pk} {type_pk}"
                    update_packaging_stock({full_name: qty_pk})
                    st.success(f"Updated {full_name}!")
                    st.rerun()

if __name__ == "__main__":
    main()