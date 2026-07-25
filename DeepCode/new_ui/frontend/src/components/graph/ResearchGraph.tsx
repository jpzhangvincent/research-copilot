import { useMemo } from 'react';
import ReactFlow, {
  Node,
  Edge,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  Handle,
  Position,
  NodeProps,
  MarkerType,
} from 'reactflow';
import 'reactflow/dist/style.css';
import type { KnowledgeGraph, GraphNode, GraphNodeType } from '../../services/api';

export const TYPE_COLORS: Record<GraphNodeType, string> = {
  paper: '#2563eb',
  method: '#7c3aed',
  dataset: '#d97706',
  metric: '#059669',
  concept: '#64748b',
  task: '#db2777',
};

interface KGNodeData {
  label: string;
  type: GraphNodeType;
  hasUrl: boolean;
  selected: boolean;
}

function KGNode({ data }: NodeProps<KGNodeData>) {
  const color = TYPE_COLORS[data.type] || TYPE_COLORS.concept;
  return (
    <div
      className="rounded-lg border px-3 py-1.5 text-xs shadow-sm transition-all"
      style={{
        borderColor: color,
        background: data.selected ? color : 'white',
        color: data.selected ? 'white' : '#1f2937',
        boxShadow: data.selected ? `0 0 0 3px ${color}44` : undefined,
        maxWidth: 190,
      }}
    >
      <Handle type="target" position={Position.Top} style={{ opacity: 0 }} />
      <div className="flex items-center gap-1.5">
        <span
          className="inline-block h-2 w-2 flex-shrink-0 rounded-full"
          style={{ background: data.selected ? 'white' : color }}
        />
        <span className="truncate font-medium">{data.label}</span>
        {data.hasUrl && <span className="flex-shrink-0 opacity-60">↗</span>}
      </div>
      <Handle type="source" position={Position.Bottom} style={{ opacity: 0 }} />
    </div>
  );
}

const nodeTypes = { kg: KGNode };

// Deterministic force-directed layout (Fruchterman–Reingold, few iterations).
function layout(graph: KnowledgeGraph): Record<string, { x: number; y: number }> {
  const nodes = graph.nodes;
  const n = nodes.length || 1;
  const W = Math.max(900, n * 55);
  const H = Math.max(600, n * 40);
  const area = W * H;
  const k = Math.sqrt(area / n) * 0.85;

  const idx: Record<string, number> = {};
  const pos = nodes.map((node, i) => {
    idx[node.id] = i;
    // seed on a circle by type for stable, non-overlapping starts
    const angle = (i / n) * Math.PI * 2;
    const r = 200 + (i % 5) * 40;
    return { x: W / 2 + Math.cos(angle) * r, y: H / 2 + Math.sin(angle) * r };
  });

  const edges = graph.edges
    .map((e) => [idx[e.source], idx[e.target]] as [number, number])
    .filter(([a, b]) => a !== undefined && b !== undefined);

  let temp = W / 8;
  const iters = 220;
  for (let it = 0; it < iters; it++) {
    const disp = pos.map(() => ({ x: 0, y: 0 }));
    // repulsion
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dx = pos[i].x - pos[j].x;
        let dy = pos[i].y - pos[j].y;
        let dist = Math.hypot(dx, dy) || 0.01;
        const force = (k * k) / dist;
        dx = (dx / dist) * force;
        dy = (dy / dist) * force;
        disp[i].x += dx; disp[i].y += dy;
        disp[j].x -= dx; disp[j].y -= dy;
      }
    }
    // attraction along edges
    for (const [a, b] of edges) {
      let dx = pos[a].x - pos[b].x;
      let dy = pos[a].y - pos[b].y;
      const dist = Math.hypot(dx, dy) || 0.01;
      const force = (dist * dist) / k;
      dx = (dx / dist) * force;
      dy = (dy / dist) * force;
      disp[a].x -= dx; disp[a].y -= dy;
      disp[b].x += dx; disp[b].y += dy;
    }
    for (let i = 0; i < n; i++) {
      const d = Math.hypot(disp[i].x, disp[i].y) || 0.01;
      pos[i].x += (disp[i].x / d) * Math.min(d, temp);
      pos[i].y += (disp[i].y / d) * Math.min(d, temp);
    }
    temp *= 0.97;
  }

  const out: Record<string, { x: number; y: number }> = {};
  nodes.forEach((node, i) => (out[node.id] = pos[i]));
  return out;
}

interface ResearchGraphProps {
  graph: KnowledgeGraph;
  onNodeClick?: (node: GraphNode) => void;
  selectedId?: string;
}

export default function ResearchGraph({ graph, onNodeClick, selectedId }: ResearchGraphProps) {
  const positions = useMemo(() => layout(graph), [graph]);

  const nodes: Node[] = useMemo(
    () =>
      graph.nodes.map((node) => ({
        id: node.id,
        type: 'kg',
        position: positions[node.id] || { x: 0, y: 0 },
        data: {
          label: node.label,
          type: node.type,
          hasUrl: !!node.url,
          selected: node.id === selectedId,
        },
      })),
    [graph, positions, selectedId]
  );

  const edges: Edge[] = useMemo(
    () =>
      graph.edges.map((e, i) => ({
        id: `${e.source}-${e.target}-${i}`,
        source: e.source,
        target: e.target,
        label: e.relation.replace(/_/g, ' '),
        labelStyle: { fontSize: 9, fill: '#9ca3af' },
        labelBgStyle: { fill: '#ffffff', fillOpacity: 0.7 },
        style: { stroke: '#d1d5db', strokeWidth: 1.2 },
        markerEnd: { type: MarkerType.ArrowClosed, color: '#d1d5db', width: 14, height: 14 },
      })),
    [graph]
  );

  const byId = useMemo(() => {
    const m: Record<string, GraphNode> = {};
    graph.nodes.forEach((n) => (m[n.id] = n));
    return m;
  }, [graph]);

  return (
    <div className="h-[600px] w-full overflow-hidden rounded-xl border border-gray-200 bg-white">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={(_, node) => onNodeClick?.(byId[node.id])}
        fitView
        minZoom={0.2}
        proOptions={{ hideAttribution: true }}
        className="bg-gray-50"
      >
        <Controls className="rounded-lg border border-gray-200 bg-white" showInteractive={false} />
        <MiniMap
          className="rounded-lg border border-gray-200 bg-white"
          pannable
          nodeColor={(node) => TYPE_COLORS[(node.data as KGNodeData).type] || '#64748b'}
        />
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#e5e7eb" />
      </ReactFlow>
    </div>
  );
}
