import streamlit as st
import folium
import pickle
import requests
import networkx as nx
import pandas as pd
from streamlit_folium import st_folium
from shapely.geometry import LineString
from scipy.spatial import KDTree
from branca.element import MacroElement
from jinja2 import Template
from pyproj import Transformer
import base64

# ========== 系統參數 ==========
map_center = [25.04, 121.56]  # 台北市中心

# ========== 關閉雙擊放大 ==========
class DisableDoubleClickZoom(MacroElement):
    def __init__(self):
        super().__init__()
        self._template = Template("""
            {% macro script(this, kwargs) %}
                {{this._parent.get_name()}}.doubleClickZoom.disable();
            {% endmacro %}
        """)

# ========== 讀取圖 ==========
@st.cache_resource
def load_graph():
    pkl_path = r"data/Tai_Road_濃度_最大連通版.pkl"
    with open(pkl_path, "rb") as f:
        G = pickle.load(f)

    transformer = Transformer.from_crs("epsg:3826", "epsg:4326", always_xy=True)
    mapping = {}
    for node in list(G.nodes):
        lon, lat = transformer.transform(node[0], node[1])
        mapping[(lat, lon)] = node
        G.nodes[node]["latlon"] = (lat, lon)

    G.graph['latlon_nodes'] = list(mapping.keys())
    G.graph['node_lookup'] = mapping
    return G

# ====== Google Geocoding ======
def geocode(address):
    api_key = "AIzaSyDnbTu8PgUkue5A9uO5aJa3lHZuNUwj6z0"
    full_address = "台灣 " + address
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": full_address, "language": "zh-TW", "key": api_key}
    try:
        response = requests.get(url, params=params).json()
        if response["status"] == "OK":
            location = response["results"][0]["geometry"]["location"]
            return location["lat"], location["lng"]
        else:
            st.warning(f"⚠️ Google 回應：{response['status']} - {response.get('error_message', '無錯誤訊息')}")
            return None
    except Exception as e:
        st.error(f"地址查詢失敗: {e}")
        return None


# ====== Reverse Geocoding（從座標查地址）======
def reverse_geocode(lat, lon):
    api_key = "AIzaSyDnbTu8PgUkue5A9uO5aJa3lHZuNUwj6z0"
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"latlng": f"{lat},{lon}", "language": "zh-TW", "key": api_key}
    try:
        response = requests.get(url, params=params).json()
        if response["status"] == "OK":
            return response["results"][0]["formatted_address"]
        else:
            return ""
    except Exception as e:
        return ""


# ========== 找最近節點 ==========
def find_nearest_node(G, lat, lon, max_dist=0.01):
    kdtree = KDTree(G.graph['latlon_nodes'])
    dist, idx = kdtree.query((lat, lon))
    if dist > max_dist:
        return None
    latlon = G.graph['latlon_nodes'][idx]
    return G.graph['node_lookup'][latlon]

# ========== 路徑計算 ==========
def compute_path(G, start_node, end_node, weight):
    try:
        path = nx.shortest_path(G, start_node, end_node, weight=lambda u, v, d: d.get("attr_dict", {}).get(weight, 0))
    except nx.NetworkXNoPath:
        return None, 0, 0

    total = 0
    exposure = 0
    for u, v in zip(path[:-1], path[1:]):
        edge_data = G.get_edge_data(u, v)
        if edge_data and "attr_dict" in edge_data:
            attrs = edge_data["attr_dict"]
            total += attrs.get("length", 0)
            exposure += attrs.get("exposure", 0)
        else:
            for d in edge_data.values():
                attrs = d.get("attr_dict", {})
                total += attrs.get("length", 0)
                exposure += attrs.get("exposure", 0)

    return path, total, exposure



################################## Streamlit 介面 ##################################
st.set_page_config(layout="wide")

# 初始化狀態（放這裡最安全）
if "show_pm25_layer" not in st.session_state:
    st.session_state.show_pm25_layer = False

# ==== 自訂按鈕樣式（可選）====
st.markdown("""
    <style>
    button[kind="primary"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 0.4em 1em !important;
    }
    </style>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns([6, 0.5, 6, 1])

with col1:
    # 標題
    st.markdown("""
        <h1 style="
            font-family: 'Noto Sans TC', 'PingFang TC', 'Microsoft JhengHei', sans-serif;
            font-size: 32px;
            font-weight: 800;
            letter-spacing: 1.5px;
            color: black;
            text-align: center;
            margin-bottom: 0px;
            line-height: 1.2;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        ">
            Geo-AI 路徑好空氣<br>
            <span style="
                font-size: 16px;
                font-weight: 500;
                color: #666666;
            ">
                台北市 & 新北市<br>
            </span>
        </h1>
    """, unsafe_allow_html=True)

    if "transport_mode" not in st.session_state:
        st.session_state.transport_mode = "機車"

    G = load_graph()
    if "points" not in st.session_state: st.session_state.points = []
    if "nodes" not in st.session_state: st.session_state.nodes = []

    # 輸入地址 / 於地圖雙擊
    st.markdown(
        """
        <style>
        .input-header-box {
            font-family: "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", sans-serif !important;
            font-size: 18px;
            font-weight: 600;
            color: #444444;
            text-align: center;
            background-color: #eeeeee;
            padding: 10px 16px;
            border-radius: 10px;
            width: 100%;
            display: block;
            margin-bottom: 12px;
        }
        .input-header-box span {
            font-size: 0.8em;
            color: #999999;
        }
        </style>

        <div class="input-header-box">
            輸入地址 / 於地圖雙擊
        </div>
        """,
        unsafe_allow_html=True
    )


    # 地址輸入框
    row1 = st.columns([1, 1])
    with row1[0]:
        if "set_start_address" in st.session_state:
            st.session_state.start_address = st.session_state.pop("set_start_address")
        start_address = st.text_input(label="", placeholder="起點地址", key="start_address")
    with row1[1]:
        if "set_end_address" in st.session_state:
            st.session_state.end_address = st.session_state.pop("set_end_address")
        end_address = st.text_input(label="", placeholder="終點地址", key="end_address")

    # 按鈕
    row2 = st.columns([3, 1, 1, 1])
    with row2[0]:
        st.markdown("""
        <style>
        input[type="radio"] {
            transform: scale(1.5);
            margin-right: 8px;
        }
        label {
            font-size: 20px;
            font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
            font-weight: 600;
        }
        </style>

        <label><input type="radio" name="mode" value="機車">機車</label>
        <label><input type="radio" name="mode" value="單車">單車</label>
        <label><input type="radio" name="mode" value="步行">步行</label>
        """, unsafe_allow_html=True)


    with row2[1]:
        # st.markdown("<div style='padding-top: 0px;'>", unsafe_allow_html=True)  # 手動對齊
        if st.button("🟢 確定起點"):
            if start_address.strip():
                result = geocode(start_address)
                if result:
                    lat, lon = result
                    nearest_node = find_nearest_node(G, lat, lon)
                    if nearest_node:
                        lat_, lon_ = G.nodes[nearest_node]["latlon"]
                        st.session_state.points = [[lat_, lon_]]
                        st.session_state.nodes = [nearest_node]
                        st.rerun()
                    else:
                        st.warning("⚠️ 離路網太遠")
            else:
                st.warning("請輸入起點地址")
        st.markdown("</div>", unsafe_allow_html=True)

    with row2[2]:
        # st.markdown("<div style='padding-top: 0x;'>", unsafe_allow_html=True)  # 手動對齊
        if st.button("🔴 確定終點"):
            if end_address.strip():
                result = geocode(end_address)
                if result:
                    lat, lon = result
                    nearest_node = find_nearest_node(G, lat, lon)
                    if nearest_node:
                        lat_, lon_ = G.nodes[nearest_node]["latlon"]
                        if len(st.session_state.points) == 1:
                            st.session_state.points.append([lat_, lon_])
                            st.session_state.nodes.append(nearest_node)
                            st.rerun()
                        else:
                            st.warning("請先設定起點")
            else:
                st.warning("請輸入終點地址")
        st.markdown("</div>", unsafe_allow_html=True)

    with row2[3]:
        if st.button("🔃 清空選擇"):
            st.session_state.points = []
            st.session_state.nodes = []
            st.rerun()


    st.markdown("""
        <style>
        div[data-testid="stForm"] {
            padding: 0 !important;
            background-color: transparent !important;
            box-shadow: none !important;
            border: none !important;
        }
        button.pm25-toggle {
            border: 2px solid #cccccc;
            border-radius: 8px;
            padding: 6px 14px;
            font-size: 16px;
            color: black;
            background-color: white;
        }
        button.pm25-toggle.active {
            border-color: red !important;
        }
        </style>
    """, unsafe_allow_html=True)


    # 統計表格
    table_row = st.columns([4,1])
    with table_row[0]:
            
        transport_mode = st.session_state.transport_mode
        SPEED = {"機車": 50, "單車": 18, "步行": 5}[transport_mode]

        if len(st.session_state.nodes) == 2:
            path1, dist1, expo1 = compute_path(G, *st.session_state.nodes, "length")
            path2, dist2, expo2 = compute_path(G, *st.session_state.nodes, "exposure")
            dist_km1, dist_km2 = dist1 / 1000, dist2 / 1000
            time_min1 = (dist_km1 / SPEED) * 60
            time_min2 = (dist_km2 / SPEED) * 60
            expo_rate1 = expo1 / time_min1 if time_min1 else 0
            expo_rate2 = expo2 / time_min2 if time_min2 else 0

            df = pd.DataFrame({
                "路徑": ["最短路徑", "最低暴露路徑"],
                "總距離(km)": [round(dist_km1, 2), round(dist_km2, 2)],
                "預估時間(min)": [round(time_min1, 2), round(time_min2, 2)],
                "每分鐘暴露量 (μg/m3)": [round(expo_rate1, 2), round(expo_rate2, 2)]
            })

            if expo1 > 0:
                improve = (expo_rate1 - expo_rate2) / expo_rate1 * 100
                st.markdown(
                    f"""
                    <div style='margin-top: 0.1em;'>
                        <h3 style='
                            font-family: "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", sans-serif !important;
                            font-size: 20px;
                            font-weight: 600;
                            color: #444444;
                            text-align: center;
                        '>
                            統計結果：改善率 {improve:.1f}%
                        </h3>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            
            st.markdown(
                f"<div class='table-wrapper'>{df.set_index('路徑').T.to_html(classes='centered-table', border=0)}</div>",
                unsafe_allow_html=True
            )


            # 加入 CSS：保留圓角邊框、移除內部格線、維持白字與透明背景（整體縮小一點）
            st.markdown("""
                <style>
                .table-wrapper {
                    width: 90%;
                    margin: auto;
                    border-radius: 12px;
                    border: 1px solid #ccc;
                    overflow: hidden;
                }
                .centered-table {
                    font-size: 14px;  /* 原本是 18px，改小一點 */
                    text-align: center;
                    width: 100%;
                    border-collapse: collapse;
                    font-family: "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", sans-serif !important;
                    background-color: transparent;
                }
                .centered-table th, .centered-table td {
                    padding: 6px;  /* 原本是 12px，改小一點 */
                    background-color: transparent;
                    color: black;
                    border: none;
                    text-align: center;
                }
                .centered-table th {
                    font-weight: bold;
                    color: black;
                    text-align: center;
                }
                .centered-table tr:hover {
                    background-color: transparent !important;
                }
                </style>
            """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <style>
        .transport-wrapper {
            background-color: #bbbbbb;
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-bottom: 20px;
            box-shadow: 1px 1px 4px rgba(0,0,0,0.05);
        }
        .transport-button {
            font-size: 14px !important;
            padding: 6px 20px !important;
            margin: 4px 0;
            width: 120px;
            text-align: center;
        }
        .legend-wrapper {
            margin-top: 0.5em;
            text-align: center;
        }
        .legend-label {
            font-size: 14px;
            font-weight: 600;
            font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
            margin: 4px 0;
            padding: 6px 16px;
            border-radius: 8px;
            background-color: #eeeeee;
            display: inline-block;
        }
        </style>

        <div class="transport-wrapper">
    """, unsafe_allow_html=True)

    map_row = st.columns([1.5, 9])
    
    with map_row[0]:

        st.markdown("""
            <style>
            .full-width-button {
                width: 100%;
                font-size: 14px !important;
                padding: 8px 0 !important;
                margin-bottom: 10px;
                text-align: center;
                font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
                font-weight: 600;
            }
            .legend-wrapper {
                margin-top: 16px;
                text-align: center;
                width: 100%;
            }
            .legend-label {
                font-size: 14px;
                font-weight: 600;
                font-family: 'Noto Sans TC', 'Microsoft JhengHei', sans-serif;
                margin: 6px auto;
                padding: 10px 0;
                border-radius: 8px;
                background-color: #eeeeee;
                display: block;
                width: 100%;
                line-height: 1.4;
            }
            </style>
        """, unsafe_allow_html=True)

        ################
        st.markdown("</div>", unsafe_allow_html=True)  # 關掉 transport-wrapper

        # 🟣 PM2.5 按鈕（獨立放置）
        with st.form(key="pm25_form"):
            submitted = st.form_submit_button("空汙疊圖")
            st.markdown(f"""
                <script>
                const btn = window.parent.document.querySelectorAll('button');
                btn.forEach(b => {{
                    if (b.innerText.includes('空汙疊圖')) {{
                        b.classList.add('full-width-button');
                        b.classList.toggle('active', {str(st.session_state.show_pm25_layer).lower()});
                    }}
                }});
                </script>
            """, unsafe_allow_html=True)
            if submitted:
                st.session_state.show_pm25_layer = not st.session_state.show_pm25_layer

        # 圖例：不可點擊的樣式展示（縮小空白）
        st.markdown("""
            <div class="legend-wrapper">
                <div class="legend-label">🟧<br>低暴路徑</div>
                <div class="legend-label">🟦<br>最短路徑</div>
            </div>
        """, unsafe_allow_html=True)




    with map_row[1]:
        
        m = folium.Map(location=map_center, zoom_start=13, control_scale=True)
        m.add_child(DisableDoubleClickZoom())

        for i, pt in enumerate(st.session_state.points):
            label = "起點" if i == 0 else "終點"
            color = "green" if i == 0 else "red"
            folium.Marker(location=pt, tooltip=label, icon=folium.Icon(color=color)).add_to(m)

        if len(st.session_state.nodes) == 2:
            for path, color, label in [
                (compute_path(G, *st.session_state.nodes, "length")[0], "blue", "最短路徑"),
                (compute_path(G, *st.session_state.nodes, "exposure")[0], "orange", "最低暴露路徑")
            ]:
                for u, v in zip(path[:-1], path[1:]):
                    edge_data = G.get_edge_data(u, v)
                    if edge_data:
                        for d in edge_data.values():
                            geom = d.get("attr_dict", {}).get("geometry")
                            if geom:
                                coords = [(lat, lon) for lon, lat in geom.coords]
                                folium.PolyLine(coords, color=color, weight=4, tooltip=label).add_to(m)
                            else:
                                pt1 = G.nodes[u]["latlon"]
                                pt2 = G.nodes[v]["latlon"]
                                folium.PolyLine([pt1, pt2], color=color, weight=4, tooltip=label).add_to(m)

        # 加入 PM2.5 疊圖層（PNG）
        if st.session_state.show_pm25_layer:
            from folium.raster_layers import ImageOverlay
            import base64

            # PNG 圖片路徑
            png_path = r"data/PM25_大台北2.png"

            # TWD97 座標
            left_twd97 = 278422.218791
            right_twd97 = 351672.218791
            bottom_twd97 = 2729604.773102
            top_twd97 = 2799454.773102

            # 轉換 TWD97 (EPSG:3826) → WGS84 (EPSG:4326)
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:3826", "EPSG:4326", always_xy=True)
            left_lon, bottom_lat = transformer.transform(left_twd97, bottom_twd97)
            right_lon, top_lat = transformer.transform(right_twd97, top_twd97)

            # 圖片轉 base64
            with open(png_path, "rb") as f:
                png_base64 = base64.b64encode(f.read()).decode("utf-8")

            # 建立疊圖層
            image_url = f"data:image/png;base64,{png_base64}"
            image_bounds = [[bottom_lat, left_lon], [top_lat, right_lon]]

            ImageOverlay(
                image=image_url,
                bounds=image_bounds,
                opacity=0.5,
                interactive=False,
                cross_origin=False,
                zindex=1,
            ).add_to(m)

        st_data = st_folium(m, width=600, height=600)

        if st_data and st_data.get("last_clicked"):
            latlon = [st_data["last_clicked"]["lat"], st_data["last_clicked"]["lng"]]
            nearest_node = find_nearest_node(G, *latlon)
            if nearest_node:
                lat_, lon_ = G.nodes[nearest_node]["latlon"]
                st.session_state.nodes.append(nearest_node)
                st.session_state.points.append([lat_, lon_])

                # 🔄 加上這段：反查地址並自動填入
                address = reverse_geocode(lat_, lon_)
                if len(st.session_state.points) == 1:
                    st.session_state["set_start_address"] = address
                elif len(st.session_state.points) == 2:
                    st.session_state["set_end_address"] = address

                st.rerun()
            else:
                st.warning("⚠️ 點的位置離路網太遠，請靠近道路再試一次。")



# ======== 加入右下角名字＋兩張 logo =========
def get_image_base64(path):
    with open(path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return encoded

# 圖片路徑
logo1_path = r"logo/成大_白色垂直.jpg"
logo2_path = r"logo/實驗室_白色方形.jpg"

# 轉 base64
logo1_base64 = get_image_base64(logo1_path)
logo2_base64 = get_image_base64(logo2_path)

# 插入名字與兩張 logo（疊在右下角，文字置中 & 放大）
st.markdown(f"""
    <style>
    .logo-wrapper {{
        position: fixed;
        bottom: 20px;
        right: 20px;
        display: flex;
        flex-direction: column;
        gap: 0px;
        z-index: 1000;
        align-items: center;
        font-family: "Noto Sans TC", "PingFang TC", "Microsoft JhengHei", sans-serif !important;
    }}
    .logo-names {{
        text-align: center;
        font-size: 15px;
        color: #666666;  /* 改深一點 */
        line-height: 1.5;
        margin-bottom: 12px;
    }}
    .logo-names .title {{
        font-size: 18px;
        font-weight: bold;
        color: #444444;  /* 標題再深一點 */
        margin-top: 6px;
        margin-bottom: 4px;
    }}
    .logo-img {{
        width: 80px;
        opacity: 0.95;
    }}
    </style>
    <div class="logo-wrapper">
        <div class="logo-names">
            <div class="title">開發人員</div>
            許家瑋<br>
            林祐如<br>
            <div class="title">指導老師</div>
            吳治達<br>
        </div>
        <img class="logo-img" src="data:image/jpg;base64,{logo1_base64}">
        <img class="logo-img" src="data:image/jpg;base64,{logo2_base64}"><br><br>
    </div>
""", unsafe_allow_html=True)
