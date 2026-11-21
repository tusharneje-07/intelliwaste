import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  BarChart3,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  CircleHelp,
  Clock3,
  Database,
  FileChartColumn,
  Gauge,
  Leaf,
  MapPinned,
  Menu,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  TestTube2,
  Upload,
  Waves,
  X,
} from "lucide-react";
import { Circle, Group, Layer, Line, Rect, Stage, Text } from "react-konva";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { create } from "zustand";

const API = import.meta.env.VITE_API_URL ?? "/api";

type Tab = "dashboard" | "test" | "evaluation";
type Zone = "Residential" | "Commercial" | "Market" | "Institutional";

interface Summary {
  total_bins: number;
  historical_records: number;
  average_fill: number;
  average_filling_rate: number;
  zone_categories: number;
  date_start: string;
  date_end: string;
  model_available: boolean;
}

interface Bin {
  bin_id: string;
  zone_type: Zone;
  latest_fill: number;
  average_fill: number;
  maximum_fill: number;
  filling_rate: number;
  latest_observation: string;
  predicted_fill: number | null;
  prediction_confidence: number | null;
  time_to_full_hours: number | null;
}

interface HistoryPoint {
  timestamp: string;
  fill_level: number;
  filling_rate: number;
  collection_event: boolean;
}

interface BinDetail extends Bin {
  history: HistoryPoint[];
  last_24_hours: HistoryPoint[];
}

interface Citizen {
  citizen_id: string;
  verified_reports: number;
  correct_reports: number;
  incorrect_reports: number;
  crs: number | null;
}

interface Prediction extends Bin {
  actual_confidence: number;
  confidence: number;
  uncertainty: number;
  confidence_threshold: number;
  demo_override_active: boolean;
  requires_verification: boolean;
  decision: string;
  validation_mae: number;
}

interface FusionStep {
  title: string;
  formula: string;
  value: string;
}

interface FusionResult {
  prediction: Prediction;
  citizen: Citizen;
  lower_limit: number;
  upper_limit: number;
  citizen_fill: number;
  fusion: {
    final_fill: number;
    lstm_error: number;
    citizen_error: number;
    lstm_reliability: number;
    citizen_reliability: number;
    lstm_weight: number;
    citizen_weight: number;
  };
  initial_time_to_full_hours: number | null;
  updated_time_to_full_hours: number | null;
  initial_time_to_full_days: number | null;
  updated_time_to_full_days: number | null;
  steps: FusionStep[];
}

interface Evaluation {
  metrics: Record<string, number>;
  training_history: Array<{ loss: number; val_loss: number }>;
  model: {
    input_shape: [number | null, number, number];
    output_shape: [number | null, number];
    parameters: number;
    layers: Array<{ name: string; type: string; parameters: number }>;
  };
  status_bands: string[];
  visualizations: Record<string, string>;
}

interface AppStore {
  activeTab: Tab;
  selectedBin: string;
  setActiveTab: (tab: Tab) => void;
  setSelectedBin: (bin: string) => void;
}

const useAppStore = create<AppStore>((set) => ({
  activeTab: "dashboard",
  selectedBin: "BIN001",
  setActiveTab: (activeTab) => set({ activeTab }),
  setSelectedBin: (selectedBin) => set({ selectedBin }),
}));

async function getJson<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, options);
  const payload = (await response.json()) as T & { error?: string };
  if (!response.ok) {
    throw new Error(payload.error ?? "The API request failed");
  }
  return payload;
}

function useSummary() {
  return useQuery({ queryKey: ["summary"], queryFn: () => getJson<Summary>("/summary") });
}

function useBins() {
  return useQuery({ queryKey: ["bins"], queryFn: () => getJson<{ bins: Bin[] }>("/bins") });
}

function useBinDetail(binId: string) {
  return useQuery({
    queryKey: ["bin", binId],
    queryFn: () => getJson<BinDetail>(`/bins/${binId}`),
    enabled: Boolean(binId),
  });
}

function useCitizens() {
  return useQuery({ queryKey: ["citizens"], queryFn: () => getJson<{ citizens: Citizen[] }>("/citizens") });
}

function useEvaluation() {
  return useQuery({ queryKey: ["evaluation"], queryFn: () => getJson<Evaluation>("/evaluation") });
}

function formatPercent(value: number | null | undefined) {
  return value === null || value === undefined ? "—" : `${value.toFixed(1)}%`;
}

function formatHours(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "Not estimable";
  return `${value.toFixed(1)} h`;
}

function formatDate(value: string) {
  return new Intl.DateTimeFormat("en", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }).format(new Date(value));
}

function fillColor(value: number) {
  if (value <= 30) return "#3e9b6d";
  if (value <= 60) return "#d4a72c";
  if (value <= 80) return "#de7a37";
  return "#d9564a";
}

function zoneColor(zone: Zone) {
  return { Residential: "#6c8f7d", Commercial: "#c18b4d", Market: "#bf6a71", Institutional: "#6680a6" }[zone];
}

function App() {
  const { activeTab, setActiveTab } = useAppStore();
  const [mobileNav, setMobileNav] = useState(false);
  const summary = useSummary();
  const bins = useBins();

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? "sidebar-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Leaf size={20} strokeWidth={2.5} /></div>
          <div><strong>IntelliWaste</strong><span>research control room</span></div>
        </div>
        <div className="side-label">Workspace</div>
        <nav className="nav-list">
          <NavButton active={activeTab === "dashboard"} icon={<Gauge size={17} />} label="Dashboard" onClick={() => { setActiveTab("dashboard"); setMobileNav(false); }} />
          <NavButton active={activeTab === "test"} icon={<TestTube2 size={17} />} label="Run a test" onClick={() => { setActiveTab("test"); setMobileNav(false); }} />
          <NavButton active={activeTab === "evaluation"} icon={<FileChartColumn size={17} />} label="Model evaluation" onClick={() => { setActiveTab("evaluation"); setMobileNav(false); }} />
        </nav>
        <div className="sidebar-footer">
          <div className="mini-status"><span className="status-dot" /> Model connected</div>
          <p>Synthetic evidence workspace<br />Jan — Apr 2026</p>
        </div>
      </aside>
      <main className="main-shell">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobileNav((current) => !current)} aria-label="Toggle navigation"><Menu size={20} /></button>
          <div className="breadcrumbs"><span>IntelliWaste</span><ChevronRight size={14} /><strong>{activeTab === "dashboard" ? "Dashboard" : activeTab === "test" ? "Test workspace" : "Evaluation"}</strong></div>
          <div className="top-actions"><span className="environment-pill"><span className="status-dot" /> Synthetic environment</span><div className="avatar">TN</div></div>
        </header>
        <div className="content">
          {activeTab === "dashboard" && <Dashboard summary={summary.data} bins={bins.data?.bins ?? []} loading={summary.isLoading || bins.isLoading} />}
          {activeTab === "test" && <TestWorkspace bins={bins.data?.bins ?? []} />}
          {activeTab === "evaluation" && <EvaluationView evaluation={useEvaluation().data} />}
        </div>
      </main>
    </div>
  );
}

function NavButton({ active, icon, label, onClick }: { active: boolean; icon: React.ReactNode; label: string; onClick: () => void }) {
  return <button className={`nav-button ${active ? "active" : ""}`} onClick={onClick}>{icon}<span>{label}</span>{active && <ChevronRight size={14} className="nav-chevron" />}</button>;
}

function Dashboard({ summary, bins, loading }: { summary?: Summary; bins: Bin[]; loading: boolean }) {
  const { selectedBin, setSelectedBin } = useAppStore();
  const [search, setSearch] = useState("");
  const [zone, setZone] = useState<"All" | Zone>("All");
  const [sortHigh, setSortHigh] = useState(true);
  const detail = useBinDetail(selectedBin);
  const filtered = useMemo(() => bins.filter((bin) => bin.bin_id.toLowerCase().includes(search.toLowerCase()) && (zone === "All" || bin.zone_type === zone)).sort((a, b) => sortHigh ? b.latest_fill - a.latest_fill : a.latest_fill - b.latest_fill), [bins, search, sortHigh, zone]);

  if (loading) return <LoadingState label="Loading the city model..." />;
  if (!summary) return <ErrorState />;
  return <PageFrame eyebrow="Operational overview" title="City waste pulse" intro="A clear read on the synthetic city’s fill levels, model signal, and collection rhythm." action={<span className="data-window"><Activity size={15} /> {formatDate(summary.date_start)} — {formatDate(summary.date_end)}</span>}>
    <div className="metric-grid">
      <MetricCard icon={<Database size={18} />} label="Total bins" value={summary.total_bins.toString()} meta="Across four zones" accent="sage" />
      <MetricCard icon={<Waves size={18} />} label="Historical records" value={summary.historical_records.toLocaleString()} meta="Hourly observations" accent="blue" />
      <MetricCard icon={<Gauge size={18} />} label="Average fill level" value={formatPercent(summary.average_fill)} meta="Across all observations" accent="amber" />
      <MetricCard icon={<Clock3 size={18} />} label="Avg. filling rate" value={`${summary.average_filling_rate.toFixed(3)}%`} meta="Percentage points / hour" accent="coral" />
    </div>
    <div className="dashboard-grid">
      <section className="panel map-panel">
        <PanelHeading icon={<MapPinned size={17} />} title="Illustrative city map" meta="Schematic placement — not geographic coordinates" />
        <CityMap bins={bins} selectedBin={selectedBin} onSelect={setSelectedBin} />
      </section>
      <section className="panel selected-panel">
        <PanelHeading icon={<Sparkles size={17} />} title={selectedBin} meta="Selected bin profile" />
        {detail.data ? <BinSnapshot detail={detail.data} /> : <LoadingState label="Loading bin detail..." compact />}
      </section>
    </div>
    <section className="panel table-panel">
      <div className="table-toolbar">
        <PanelHeading icon={<Database size={17} />} title="Bin register" meta={`${filtered.length} of ${bins.length} bins`} />
        <div className="toolbar-controls">
          <label className="search-input"><Search size={15} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search bin ID" /></label>
          <select value={zone} onChange={(event) => setZone(event.target.value as "All" | Zone)}><option>All</option>{(["Residential", "Commercial", "Market", "Institutional"] as Zone[]).map((value) => <option key={value}>{value}</option>)}</select>
          <button className="ghost-button" onClick={() => setSortHigh((current) => !current)}>{sortHigh ? "Highest fill" : "Lowest fill"} <ArrowUpRight size={14} /></button>
        </div>
      </div>
      <BinTable bins={filtered} onSelect={setSelectedBin} />
    </section>
    {detail.data && <section className="panel history-panel"><PanelHeading icon={<Activity size={17} />} title={`${selectedBin} history`} meta="Hourly fill level with collection events" /><HistoryChart detail={detail.data} /></section>}
  </PageFrame>;
}

function CityMap({ bins, selectedBin, onSelect }: { bins: Bin[]; selectedBin: string; onSelect: (binId: string) => void }) {
  const wrapper = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(820);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  useEffect(() => {
    if (!wrapper.current) return;
    const observer = new ResizeObserver((entries) => setWidth(Math.max(620, Math.floor(entries[0].contentRect.width))));
    observer.observe(wrapper.current);
    return () => observer.disconnect();
  }, []);
  const mapWidth = Math.max(620, width);
  const scale = mapWidth / 820;
  const mapHeight = Math.round(650 * scale);
  const positions: Record<string, [number, number]> = { BIN001: [125, 135], BIN002: [260, 185], BIN003: [175, 300], BIN004: [465, 125], BIN005: [625, 180], BIN006: [530, 285], BIN007: [170, 500], BIN008: [350, 445], BIN009: [500, 505], BIN010: [690, 345], BIN011: [730, 470], BIN012: [650, 575] };
  const horizontalRoads = [75, 153, 238, 330, 420, 516, 600];
  const verticalRoads = [94, 190, 292, 395, 500, 610, 724];
  const buildings = [
    [82, 35, 30, 24], [135, 42, 50, 31], [220, 31, 36, 25], [275, 49, 63, 30], [355, 32, 26, 44],
    [83, 101, 46, 29], [151, 105, 28, 37], [232, 92, 58, 33], [310, 103, 34, 25], [349, 96, 45, 25],
    [83, 181, 35, 37], [139, 174, 44, 24], [216, 169, 31, 48], [256, 180, 53, 27], [325, 171, 34, 40],
    [440, 43, 46, 27], [506, 34, 33, 40], [555, 51, 56, 25], [635, 35, 36, 33], [686, 50, 48, 29],
    [439, 107, 29, 28], [480, 98, 64, 35], [575, 104, 33, 37], [623, 96, 57, 30], [700, 111, 32, 26],
    [440, 180, 58, 29], [522, 171, 31, 47], [569, 187, 60, 24], [648, 172, 38, 40], [704, 185, 35, 27],
    [92, 372, 54, 31], [164, 360, 31, 45], [221, 377, 65, 24], [312, 365, 39, 34], [350, 382, 27, 45],
    [90, 449, 32, 25], [137, 438, 60, 34], [226, 447, 36, 27], [282, 431, 54, 45], [350, 463, 37, 28],
    [442, 371, 30, 43], [485, 361, 62, 28], [579, 377, 33, 39], [630, 363, 58, 30], [707, 376, 32, 44],
    [448, 448, 54, 25], [518, 438, 35, 45], [578, 455, 66, 27], [674, 444, 41, 36], [715, 490, 27, 28],
    [86, 551, 42, 28], [151, 535, 63, 39], [240, 556, 32, 25], [295, 534, 55, 34], [365, 553, 38, 28],
    [448, 548, 38, 33], [510, 532, 63, 31], [598, 548, 31, 42], [650, 531, 59, 29], [716, 560, 29, 38],
  ];
  const changeZoom = (nextZoom: number) => {
    setZoom(Math.min(1.8, Math.max(0.75, Number(nextZoom.toFixed(2)))));
    setPan({ x: 0, y: 0 });
  };
  const baseX = 410 * (1 - zoom);
  const baseY = 325 * (1 - zoom);
  const minX = zoom >= 1 ? 820 * (1 - zoom) : 0;
  const maxX = zoom >= 1 ? 0 : 820 * (1 - zoom);
  const minY = zoom >= 1 ? 650 * (1 - zoom) : 0;
  const maxY = zoom >= 1 ? 0 : 650 * (1 - zoom);
  const dragBound = (position: { x: number; y: number }) => ({
    x: Math.min(maxX, Math.max(minX, position.x)),
    y: Math.min(maxY, Math.max(minY, position.y)),
  });
  return <div className="map-wrap" ref={wrapper}><div className="map-stage-viewport"><Stage width={mapWidth} height={mapHeight} scaleX={scale} scaleY={scale} onWheel={(event) => { event.evt.preventDefault(); changeZoom(zoom * (event.evt.deltaY > 0 ? 0.9 : 1.1)); }}>
    <Layer>
      <Group x={baseX + pan.x} y={baseY + pan.y} scaleX={zoom} scaleY={zoom} draggable={zoom !== 1} dragBoundFunc={dragBound} onDragEnd={(event) => setPan({ x: event.target.x() - baseX, y: event.target.y() - baseY })}>
      <Rect x={0} y={0} width={820} height={650} fill="#eef0ef" cornerRadius={20} />
      <Rect x={0} y={0} width={52} height={650} fill="#dce7e9" cornerRadius={20} />
      <Rect x={770} y={0} width={50} height={650} fill="#dce7e9" cornerRadius={20} />
      <Text x={8} y={252} text="RIVER EDGE" fontFamily="DM Mono" fontSize={9} fill="#7b9294" rotation={-90} />
      <Text x={780} y={250} text="CANAL" fontFamily="DM Mono" fontSize={9} fill="#7b9294" rotation={90} />
      {buildings.map(([x, y, buildingWidth, buildingHeight], index) => <Rect key={`building-${index}`} x={x} y={y} width={buildingWidth} height={buildingHeight} fill={index % 4 === 0 ? "#e0e2e1" : "#e5e7e6"} stroke="#d4d8d6" strokeWidth={1} cornerRadius={2} />)}
      {horizontalRoads.map((y) => <Group key={`h-${y}`}><Line points={[52, y, 770, y]} stroke="#ffffff" strokeWidth={15} /><Line points={[52, y, 770, y]} stroke="#d5d9d7" strokeWidth={1} /></Group>)}
      {verticalRoads.map((x) => <Group key={`v-${x}`}><Line points={[x, 0, x, 650]} stroke="#ffffff" strokeWidth={13} /><Line points={[x, 0, x, 650]} stroke="#d7dbd9" strokeWidth={1} /></Group>)}
      <Group><Line points={[52, 570, 185, 453, 338, 337, 474, 260, 770, 75]} stroke="#ffffff" strokeWidth={22} /><Line points={[52, 570, 185, 453, 338, 337, 474, 260, 770, 75]} stroke="#cdd2cf" strokeWidth={1.4} /></Group>
      <Group><Line points={[52, 95, 180, 164, 320, 220, 478, 275, 770, 585]} stroke="#ffffff" strokeWidth={12} /><Line points={[52, 95, 180, 164, 320, 220, 478, 275, 770, 585]} stroke="#d2d7d4" strokeWidth={1} /></Group>
      {[{ x: 395, y: 244 }, { x: 190, y: 453 }, { x: 610, y: 330 }].map((roundabout) => <Group key={`${roundabout.x}-${roundabout.y}`}><Circle x={roundabout.x} y={roundabout.y} radius={21} fill="#eef0ef" stroke="#ffffff" strokeWidth={8} /><Circle x={roundabout.x} y={roundabout.y} radius={9} fill="#e1e4e1" stroke="#cbd0cd" strokeWidth={1} /></Group>)}
      <Rect x={122} y={265} width={76} height={50} fill="#d5e1d6" stroke="#c4d2c5" strokeWidth={1} cornerRadius={6} />
      <Rect x={571} y={444} width={112} height={67} fill="#d5e1d6" stroke="#c4d2c5" strokeWidth={1} cornerRadius={6} />
      <Text x={136} y={286} text="NORTH PARK" fontFamily="DM Mono" fontSize={8} fill="#69806c" />
      <Text x={588} y={472} text="CIVIC GREEN" fontFamily="DM Mono" fontSize={8} fill="#69806c" />
      <Text x={98} y={60} text="SHIVAJI ROAD" fontFamily="DM Mono" fontSize={10} fill="#777f7b" rotation={8} />
      <Text x={294} y={197} text="RAJARAM ROAD" fontFamily="DM Mono" fontSize={10} fill="#777f7b" />
      <Text x={588} y={137} text="CENTRAL MARKET" fontFamily="DM Mono" fontSize={9} fill="#818984" rotation={-2} />
      <Text x={458} y={555} text="CIVIC CAMPUS" fontFamily="DM Mono" fontSize={9} fill="#818984" />
      <Text x={340} y={617} text="Illustrative street layer • no GPS data" fontFamily="DM Mono" fontSize={9} fill="#78837d" />
      {bins.map((bin) => {
        const [x, y] = positions[bin.bin_id] ?? [410, 320];
        const selected = bin.bin_id === selectedBin;
        return <Group key={bin.bin_id} x={x} y={y} onClick={() => onSelect(bin.bin_id)} onTap={() => onSelect(bin.bin_id)}>
          <Circle radius={selected ? 21 : 17} fill="#ffffff" opacity={0.84} />
          <Circle radius={selected ? 14 : 11} fill={fillColor(bin.latest_fill)} stroke={selected ? "#8b6d1d" : "#ffffff"} strokeWidth={selected ? 3 : 2} shadowColor="#6d5b27" shadowBlur={selected ? 10 : 5} shadowOpacity={0.28} />
          <Rect x={-5} y={-4} width={10} height={10} fill="#39443e" cornerRadius={2} />
          <Rect x={-6} y={-7} width={12} height={2} fill="#39443e" cornerRadius={1} />
          <Line points={[-2, -1, -2, 4, 2, 4, 2, -1]} stroke="#f6d46b" strokeWidth={1} />
          <Rect x={-25} y={17} width={50} height={14} fill="#ffffffdd" cornerRadius={4} />
          <Text text={bin.bin_id} fontFamily="DM Mono" fontSize={7} fill="#345044" align="center" width={50} x={-25} y={20} />
        </Group>;
      })}
      </Group>
    </Layer>
  </Stage></div>
    <div className="map-controls" aria-label="Map zoom controls">
      <button onClick={() => changeZoom(zoom + 0.15)} aria-label="Zoom in">+</button>
      <span>{Math.round(zoom * 100)}%</span>
      <button onClick={() => changeZoom(zoom - 0.15)} aria-label="Zoom out">−</button>
      <button onClick={() => changeZoom(1)} aria-label="Reset map zoom">Reset</button>
    </div>
    <div className="map-legend">{[["0–30%", "#3e9b6d"], ["30–60%", "#d4a72c"], ["60–80%", "#de7a37"], ["80–100%", "#d9564a"]].map(([label, color]) => <span key={label}><i style={{ background: color }} />{label}</span>)}</div>
  </div>;
}

function ZoneShape({ x, y, width, height, color, label }: { x: number; y: number; width: number; height: number; color: string; label: string }) {
  return <Group><Rect x={x} y={y} width={width} height={height} fill={color} cornerRadius={18} /><Text x={x + 18} y={y + 18} text={label.toUpperCase()} fontFamily="DM Mono" fontSize={11} fill="#436153" letterSpacing={1.2} /></Group>;
}

function BinSnapshot({ detail }: { detail: BinDetail }) {
  const last = detail.history[detail.history.length - 1];
  return <div className="snapshot">
    <div className="snapshot-hero"><div><span className="eyebrow">{detail.zone_type} zone</span><strong>{formatPercent(detail.latest_fill)}</strong><small>Latest recorded fill</small></div><div className="fill-ring" style={{ "--fill": `${detail.latest_fill}%`, "--ring-color": fillColor(detail.latest_fill) } as React.CSSProperties}><span>{Math.round(detail.latest_fill)}</span></div></div>
    <div className="mini-stat-grid"><MiniStat label="Average fill" value={formatPercent(detail.average_fill)} /><MiniStat label="Peak fill" value={formatPercent(detail.maximum_fill)} /><MiniStat label="Rate" value={`${detail.filling_rate.toFixed(3)}% / h`} /><MiniStat label="Next hour" value={formatPercent(detail.predicted_fill)} /></div>
    <div className="snapshot-foot"><span>Last observed</span><strong>{formatDate(last.timestamp)}</strong><span>Time-to-full</span><strong>{formatHours(detail.time_to_full_hours)}</strong></div>
  </div>;
}

function BinTable({ bins, onSelect }: { bins: Bin[]; onSelect: (binId: string) => void }) {
  return <div className="data-table-wrap"><table className="data-table"><thead><tr><th>Bin</th><th>Zone</th><th>Latest fill</th><th>Average</th><th>Rate / hour</th><th>Predicted</th><th>Time-to-full</th><th /></tr></thead><tbody>{bins.map((bin) => <tr key={bin.bin_id} onClick={() => onSelect(bin.bin_id)}><td><span className="bin-code">{bin.bin_id}</span></td><td><span className="zone-tag" style={{ "--zone": zoneColor(bin.zone_type) } as React.CSSProperties}>{bin.zone_type}</span></td><td><span className="fill-value" style={{ color: fillColor(bin.latest_fill) }}>{formatPercent(bin.latest_fill)}</span></td><td>{formatPercent(bin.average_fill)}</td><td>{bin.filling_rate.toFixed(3)}%</td><td>{formatPercent(bin.predicted_fill)}</td><td>{formatHours(bin.time_to_full_hours)}</td><td><ChevronRight size={15} /></td></tr>)}</tbody></table></div>;
}

function HistoryChart({ detail }: { detail: BinDetail }) {
  const data = detail.history.filter((_, index) => index % 12 === 0 || index === detail.history.length - 1).map((point) => ({ time: new Date(point.timestamp).toLocaleDateString("en", { month: "short", day: "numeric" }), fill: point.fill_level, rate: point.filling_rate }));
  return <div className="chart-large"><ResponsiveContainer width="100%" height={260}><AreaChart data={data}><defs><linearGradient id="fillWash" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#5a9d73" stopOpacity={0.34} /><stop offset="100%" stopColor="#5a9d73" stopOpacity={0.02} /></linearGradient></defs><CartesianGrid stroke="#e8eee8" vertical={false} /><XAxis dataKey="time" tickLine={false} axisLine={false} minTickGap={24} /><YAxis domain={[0, 100]} tickLine={false} axisLine={false} unit="%" /><Tooltip formatter={(value, name) => { const numeric = Number(value ?? 0); const key = String(name); return [key === "fill" ? `${numeric.toFixed(1)}%` : `${numeric.toFixed(3)}% / h`, key === "fill" ? "Fill level" : "Rate"]; }} /><Area type="monotone" dataKey="fill" stroke="#34795a" fill="url(#fillWash)" strokeWidth={2} dot={false} /></AreaChart></ResponsiveContainer></div>;
}

function TestWorkspace({ bins }: { bins: Bin[] }) {
  const { selectedBin, setSelectedBin } = useAppStore();
  const { data: citizens } = useCitizens();
  const detail = useBinDetail(selectedBin);
  const [override, setOverride] = useState(false);
  const [demoConfidence, setDemoConfidence] = useState("0.60");
  const [citizenId, setCitizenId] = useState("");
  const [lower, setLower] = useState("50");
  const [upper, setUpper] = useState("75");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const predictMutation = useMutation({ mutationFn: (body: { bin_id: string; demo_confidence?: number }) => getJson<Prediction>("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }) });
  const fusionMutation = useMutation({ mutationFn: (body: Record<string, string | number>) => getJson<FusionResult>("/fusion", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }) });
  const classifierMutation = useMutation({ mutationFn: () => getJson<{ message: string }>("/classify", { method: "POST" }) });
  const prediction = predictMutation.data;
  const selectedCitizen = citizens?.citizens.find((citizen) => citizen.citizen_id === citizenId);
  const citizenFill = (Number(lower) + Number(upper)) / 2;
  const runPrediction = () => predictMutation.mutate({ bin_id: selectedBin, ...(override ? { demo_confidence: Number(demoConfidence) } : {}) });
  useEffect(() => { if (!citizenId && citizens?.citizens[0]) setCitizenId(citizens.citizens[0].citizen_id); }, [citizenId, citizens]);
  return <PageFrame eyebrow="Interactive test bench" title="Run the full IntelliWaste workflow" intro="Select a real bin from the synthetic history, ask the existing LSTM for its next-hour estimate, then branch into citizen fusion only when confidence requires it." action={<span className="model-pill"><BrainCircuit size={15} /> Existing Keras model</span>}>
    <div className="test-layout">
      <section className="panel test-controls">
        <PanelHeading icon={<TestTube2 size={17} />} title="Prediction setup" meta="No manual history entry required" />
        <label className="field-label">Select bin<select value={selectedBin} onChange={(event) => { setSelectedBin(event.target.value); predictMutation.reset(); fusionMutation.reset(); }}><option value="">Choose a bin</option>{bins.map((bin) => <option key={bin.bin_id}>{bin.bin_id}</option>)}</select></label>
        {detail.data && <div className="input-preview"><div><span>Zone</span><strong>{detail.data.zone_type}</strong></div><div><span>Latest fill</span><strong>{formatPercent(detail.data.latest_fill)}</strong></div><div><span>Sequence</span><strong>24 hourly rows</strong></div></div>}
        <div className="override-box"><div className="toggle-row"><div><strong>Simulate low-confidence scenario</strong><span>Demo-only; does not alter model or metrics.</span></div><button className={`toggle ${override ? "on" : ""}`} onClick={() => setOverride((value) => !value)} aria-label="Toggle low-confidence simulation"><span /></button></div>{override && <label className="field-label compact">Demo confidence (0–1)<input type="number" min="0" max="1" step="0.01" value={demoConfidence} onChange={(event) => setDemoConfidence(event.target.value)} /></label>}</div>
        <button className="primary-button full-width" onClick={runPrediction} disabled={!selectedBin || predictMutation.isPending}><RefreshCw size={16} className={predictMutation.isPending ? "spin" : ""} />{predictMutation.isPending ? "Running model..." : "Run IntelliWaste prediction"}</button>
        {predictMutation.error && <div className="error-banner"><AlertTriangle size={16} />{predictMutation.error.message}</div>}
      </section>
      <section className="panel prediction-panel">
        <PanelHeading icon={<BrainCircuit size={17} />} title="Model signal" meta={prediction?.demo_override_active ? "Demo override active" : "Validation-derived confidence"} />
        {prediction ? <PredictionCard prediction={prediction} /> : <EmptyState icon={<Sparkles size={24} />} title="Ready when you are" text="Run a prediction to see the actual model output and confidence decision." />}
        {detail.data && <div className="chart-small"><ResponsiveContainer width="100%" height={190}><LineChart data={detail.data.last_24_hours.map((point) => ({ hour: new Date(point.timestamp).toLocaleTimeString("en", { hour: "2-digit" }), fill: point.fill_level }))}><CartesianGrid stroke="#e8eee8" vertical={false} /><XAxis dataKey="hour" hide /><YAxis domain={[0, 100]} tickLine={false} axisLine={false} unit="%" /><Tooltip /><Line type="monotone" dataKey="fill" stroke="#34795a" strokeWidth={2.5} dot={false} /></LineChart></ResponsiveContainer></div>}
      </section>
    </div>
    {prediction?.requires_verification && <CitizenForm citizens={citizens?.citizens ?? []} citizenId={citizenId} setCitizenId={setCitizenId} lower={lower} setLower={setLower} upper={upper} setUpper={setUpper} citizen={selectedCitizen} citizenFill={citizenFill} onSubmit={() => fusionMutation.mutate({ bin_id: selectedBin, citizen_id: citizenId, lower_limit: Number(lower), upper_limit: Number(upper), ...(override ? { demo_confidence: Number(demoConfidence) } : {}) })} pending={fusionMutation.isPending} />}
    {fusionMutation.data && <FusionSteps result={fusionMutation.data} />}
    <section className="panel classifier-panel"><PanelHeading icon={<Upload size={17} />} title="Independent waste classifier" meta="Classification only — never used for fill fusion" /><div className="classifier-row"><div><strong>Organic · Recyclable · E-waste</strong><p>Upload a citizen photo when a fine-tuned three-class checkpoint is configured.</p></div><label className="upload-button"><Upload size={15} />{uploadedFile ? uploadedFile.name : "Choose image"}<input type="file" accept="image/*" onChange={(event) => setUploadedFile(event.target.files?.[0] ?? null)} /></label><button className="ghost-button" onClick={() => classifierMutation.mutate()} disabled={classifierMutation.isPending}>Check classifier</button></div>{classifierMutation.data && <div className="info-banner"><CircleHelp size={16} />{classifierMutation.data.message}</div>}</section>
  </PageFrame>;
}

function PredictionCard({ prediction }: { prediction: Prediction }) {
  return <div className="prediction-card"><div className="prediction-main"><span className="eyebrow">Next-hour LSTM estimate</span><strong>{formatPercent(prediction.predicted_fill)}</strong><span className="confidence-chip" style={{ "--chip": prediction.confidence >= 0.8 ? "#34795a" : "#c15b43" } as React.CSSProperties}>{prediction.confidence >= 0.8 ? <CheckCircle2 size={13} /> : <AlertTriangle size={13} />}{formatPercent(prediction.confidence * 100)} confidence</span></div><div className="prediction-stats"><div><span>Uncertainty</span><strong>{formatPercent(prediction.uncertainty * 100)}</strong></div><div><span>Threshold</span><strong>{formatPercent(prediction.confidence_threshold * 100)}</strong></div><div><span>Decision</span><strong className={prediction.requires_verification ? "text-coral" : "text-sage"}>{prediction.requires_verification ? "Verification required" : "Retain LSTM"}</strong></div></div></div>;
}

function CitizenForm({ citizens, citizenId, setCitizenId, lower, setLower, upper, setUpper, citizen, citizenFill, onSubmit, pending }: { citizens: Citizen[]; citizenId: string; setCitizenId: (value: string) => void; lower: string; setLower: (value: string) => void; upper: string; setUpper: (value: string) => void; citizen?: Citizen; citizenFill: number; onSubmit: () => void; pending: boolean }) {
  return <section className="panel verification-panel"><PanelHeading icon={<ShieldCheck size={17} />} title="Citizen verification required" meta="Method A · category range selection" /><div className="verification-grid"><label className="field-label">Select citizen<select value={citizenId} onChange={(event) => setCitizenId(event.target.value)}>{citizens.map((item) => <option key={item.citizen_id}>{item.citizen_id}</option>)}</select></label><label className="field-label">Lower fill limit<select value={lower} onChange={(event) => setLower(event.target.value)}>{["0", "25", "50", "75", "100"].map((value) => <option key={value} value={value}>{value}%</option>)}</select></label><label className="field-label">Upper fill limit<select value={upper} onChange={(event) => setUpper(event.target.value)}>{["0", "25", "50", "75", "100"].map((value) => <option key={value} value={value}>{value}%</option>)}</select></label></div><div className="citizen-preview"><div><span>Citizen fill estimate</span><strong>{citizenFill.toFixed(1)}%</strong><small>Fc = (Lc + Uc) / 2</small></div><div><span>Historical reliability</span><strong>{citizen?.crs === null || citizen?.crs === undefined ? "Unavailable" : `${(citizen.crs * 100).toFixed(1)}%`}</strong><small>{citizen?.correct_reports ?? 0} correct / {citizen?.verified_reports ?? 0} verified</small></div><button className="primary-button" onClick={onSubmit} disabled={pending || Number(lower) > Number(upper)}><ShieldCheck size={16} />{pending ? "Calculating..." : "Submit verification"}</button></div>{Number(lower) > Number(upper) && <div className="error-banner"><AlertTriangle size={16} />Lower limit cannot exceed upper limit.</div>}</section>;
}

function FusionSteps({ result }: { result: FusionResult }) {
  return <section className="panel fusion-panel"><div className="fusion-header"><PanelHeading icon={<Sparkles size={17} />} title="Step-by-step fusion result" meta="LSTM + citizen report only" /><div className="final-result"><span>Final fill</span><strong>{formatPercent(result.fusion.final_fill)}</strong></div></div><div className="step-grid">{result.steps.map((step, index) => <details className="calculation-step" key={step.title} open><summary><span className="step-number">{String(index + 1).padStart(2, "0")}</span><strong>{step.title}</strong><ChevronRight size={15} /></summary><div><code>{step.formula}</code><p>{step.value}</p></div></details>)}</div><div className="before-after"><div><span>Initial LSTM time-to-full</span><strong>{formatHours(result.initial_time_to_full_hours)} <small>{result.initial_time_to_full_days ? `(${result.initial_time_to_full_days.toFixed(1)} days)` : ""}</small></strong></div><ChevronRight size={18} /><div><span>Updated fused time-to-full</span><strong>{formatHours(result.updated_time_to_full_hours)} <small>{result.updated_time_to_full_days ? `(${result.updated_time_to_full_days.toFixed(1)} days)` : ""}</small></strong></div></div></section>;
}

function EvaluationView({ evaluation }: { evaluation?: Evaluation }) {
  if (!evaluation) return <LoadingState label="Loading evaluation artifacts..." />;
  const metrics = [{ label: "Test MAE", value: `${evaluation.metrics.test_mae?.toFixed(3) ?? "—"} pp`, icon: <Activity size={16} /> }, { label: "Test RMSE", value: `${evaluation.metrics.test_rmse?.toFixed(3) ?? "—"} pp`, icon: <Waves size={16} /> }, { label: "Band accuracy", value: `${((evaluation.metrics.test_accuracy ?? 0) * 100).toFixed(1)}%`, icon: <CheckCircle2 size={16} /> }, { label: "Macro F1", value: `${((evaluation.metrics.test_f1_macro ?? 0) * 100).toFixed(1)}%`, icon: <ShieldCheck size={16} /> }];
  const curve = evaluation.training_history.map((point, index) => ({ epoch: index + 1, loss: point.loss, validation: point.val_loss }));
  const image = (name: string) => `${API}/artifacts/${evaluation.visualizations[name]}`;
  return <PageFrame eyebrow="Evidence & diagnostics" title="Model evaluation" intro="The evaluation view keeps continuous fill regression metrics separate from the supplementary five-band status classification.">
    <div className="metric-grid">{metrics.map((metric) => <MetricCard key={metric.label} icon={metric.icon} label={metric.label} value={metric.value} meta="Independent chronological test set" accent="blue" />)}</div>
    <div className="evaluation-grid"><section className="panel chart-panel"><PanelHeading icon={<Activity size={17} />} title="Training vs validation loss" meta={`${curve.length} epochs recorded`} /><ResponsiveContainer width="100%" height={280}><LineChart data={curve}><CartesianGrid stroke="#e8eee8" vertical={false} /><XAxis dataKey="epoch" tickLine={false} axisLine={false} /><YAxis tickLine={false} axisLine={false} /><Tooltip /><Legend /><Line type="monotone" dataKey="loss" name="Training loss" stroke="#34795a" strokeWidth={2} dot={false} /><Line type="monotone" dataKey="validation" name="Validation loss" stroke="#d47c42" strokeWidth={2} dot={false} /></LineChart></ResponsiveContainer></section><section className="panel model-panel"><PanelHeading icon={<BrainCircuit size={17} />} title="Saved model" meta="Loaded from existing Keras artifact" /><div className="model-specs"><MiniStat label="Input sequence" value={`${evaluation.model.input_shape[1]} hours`} /><MiniStat label="Input features" value={String(evaluation.model.input_shape[2])} /><MiniStat label="Output shape" value={String(evaluation.model.output_shape[1])} /><MiniStat label="Parameters" value={evaluation.model.parameters.toLocaleString()} /></div><table className="layer-table"><thead><tr><th>Layer</th><th>Type</th><th>Parameters</th></tr></thead><tbody>{evaluation.model.layers.map((layer) => <tr key={layer.name}><td>{layer.name}</td><td>{layer.type}</td><td>{layer.parameters.toLocaleString()}</td></tr>)}</tbody></table></section></div>
    <section className="panel chart-panel"><PanelHeading icon={<BarChart3 size={17} />} title="Evaluation visualizations" meta="Generated from actual test artifacts" /><div className="visual-grid">{Object.entries(evaluation.visualizations).map(([name, path]) => <figure key={name}><img src={`${API}/artifacts/${path}`} alt={name.replaceAll("_", " ")} /><figcaption>{name.replaceAll("_", " ")}</figcaption></figure>)}</div></section>
    <section className="panel metric-explainer"><PanelHeading icon={<CircleHelp size={17} />} title="How to read this" meta="Methodology note" /><p>MAE and RMSE are continuous fill-percentage errors in percentage points. Accuracy, precision, recall, F1, and the confusion matrix are derived from the documented Empty / Low / Half / High / Full status bands; they are not raw percentage accuracy.</p></section>
  </PageFrame>;
}

function PageFrame({ eyebrow, title, intro, action, children }: { eyebrow: string; title: string; intro: string; action?: React.ReactNode; children: React.ReactNode }) {
  return <div className="page-frame"><div className="page-heading"><div><span className="eyebrow">{eyebrow}</span><h1>{title}</h1><p>{intro}</p></div>{action}</div>{children}</div>;
}

function PanelHeading({ icon, title, meta }: { icon: React.ReactNode; title: string; meta: string }) {
  return <div className="panel-heading"><div className="panel-title">{icon}<div><h2>{title}</h2><span>{meta}</span></div></div></div>;
}

function MetricCard({ icon, label, value, meta, accent }: { icon: React.ReactNode; label: string; value: string; meta: string; accent: "sage" | "blue" | "amber" | "coral" }) {
  return <div className={`metric-card ${accent}`}><div className="metric-icon">{icon}</div><span>{label}</span><strong>{value}</strong><small>{meta}</small></div>;
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return <div className="mini-stat"><span>{label}</span><strong>{value}</strong></div>;
}

function EmptyState({ icon, title, text }: { icon: React.ReactNode; title: string; text: string }) {
  return <div className="empty-state">{icon}<strong>{title}</strong><span>{text}</span></div>;
}

function LoadingState({ label, compact = false }: { label: string; compact?: boolean }) {
  return <div className={`loading-state ${compact ? "compact" : ""}`}><RefreshCw size={18} className="spin" /><span>{label}</span></div>;
}

function ErrorState() {
  return <div className="error-state"><AlertTriangle size={20} /><strong>Backend unavailable</strong><span>Start Flask with <code>uv run python app/backend/app.py</code>, then refresh this page.</span></div>;
}

export default App;
