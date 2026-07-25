import { useEffect, useState } from 'react';
import { useSearchParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Network, Search, Zap, Clock, DollarSign, ExternalLink, BookOpen,
  Loader2, FileText, Database, Gauge, Lightbulb, Wrench, Sparkles,
} from 'lucide-react';
import { Card, Button } from '../components/common';
import ResearchGraph, { TYPE_COLORS } from '../components/graph/ResearchGraph';
import { copilotApi } from '../services/api';
import type { KnowledgeGraph, GraphNode, GraphNodeType } from '../services/api';
import { toast } from '../components/common/Toaster';

const TYPE_META: Record<GraphNodeType, { label: string; icon: typeof FileText }> = {
  paper: { label: 'Papers', icon: FileText },
  method: { label: 'Methods', icon: Wrench },
  dataset: { label: 'Datasets', icon: Database },
  metric: { label: 'Metrics', icon: Gauge },
  concept: { label: 'Concepts', icon: Lightbulb },
  task: { label: 'Tasks', icon: Sparkles },
};

const EFFORTS = ['standard', 'deep', 'exhaustive'];

export default function KnowledgeGraphPage() {
  const [params] = useSearchParams();
  const queryTopic = params.get('topic');
  const [topic, setTopic] = useState(queryTopic || 'LLM confidence reranker for RAG');
  const [effort, setEffort] = useState('standard');
  const [graph, setGraph] = useState<KnowledgeGraph | null>(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [compilingUrl, setCompilingUrl] = useState<string | null>(null);

  // Load a cached graph on first render for an instant payoff (topic-specific if provided).
  useEffect(() => {
    copilotApi
      .getCachedGraph(queryTopic || undefined)
      .then((g) => {
        setGraph(g);
        if (!queryTopic) setTopic(g.topic || 'LLM confidence reranker for RAG');
      })
      .catch(() => {/* no cache yet — user can research live */});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const research = async () => {
    if (!topic.trim()) return;
    setLoading(true);
    setSelected(null);
    try {
      const g = await copilotApi.buildGraph(topic.trim(), null, effort);
      setGraph(g);
      toast.success('Graph built', `${g.nodes.length} nodes · ${g.edges.length} edges`);
    } catch {
      toast.error('Research failed', 'Could not reach You.com. Is the backend up?');
    } finally {
      setLoading(false);
    }
  };

  const loadCached = async () => {
    try {
      const g = await copilotApi.getCachedGraph(topic.trim() || undefined);
      setGraph(g);
      setSelected(null);
    } catch {
      toast.info('No cached graph', 'Run live research to build one.');
    }
  };

  const summarize = async (node: GraphNode) => {
    if (!node.url) return;
    setCompilingUrl(node.url);
    try {
      const res = await copilotApi.compileWiki(node.url, 'llm-agents', graph?.topic);
      toast.success('Added to wiki', res.rel_path);
    } catch {
      toast.error('Librarian failed', 'Could not compile this paper.');
    } finally {
      setCompilingUrl(null);
    }
  };

  const counts = graph
    ? graph.nodes.reduce<Record<string, number>>((acc, n) => {
        acc[n.type] = (acc[n.type] || 0) + 1;
        return acc;
      }, {})
    : {};

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-2">
          <Network className="h-6 w-6 text-primary-600" />
          <h1 className="text-2xl font-bold text-gray-900">Knowledge Graph</h1>
        </div>
        <p className="mt-1 text-gray-500">
          The Cartographer runs multi-step You.com research and maps the landscape —
          papers, methods, datasets, metrics and how they connect.
        </p>
      </motion.div>

      {/* Controls */}
      <Card>
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center">
          <div className="flex flex-1 items-center gap-2">
            <Search className="h-4 w-4 flex-shrink-0 text-gray-400" />
            <input
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && research()}
              placeholder="A research area to map…"
              className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
            />
          </div>
          <select
            value={effort}
            onChange={(e) => setEffort(e.target.value)}
            className="rounded-lg border border-gray-300 px-3 py-2 text-sm text-gray-700 focus:border-primary-500 focus:outline-none"
            title="You.com research effort"
          >
            {EFFORTS.map((e) => (
              <option key={e} value={e}>{e}</option>
            ))}
          </select>
          <div className="flex gap-2">
            <Button variant="secondary" onClick={loadCached} disabled={loading}>
              <Zap className="mr-2 h-4 w-4" /> Cached
            </Button>
            <Button onClick={research} isLoading={loading}>
              <Network className="mr-2 h-4 w-4" /> Research live
            </Button>
          </div>
        </div>

        {graph && (
          <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
            <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 font-medium ${graph.cached ? 'bg-amber-50 text-amber-700' : 'bg-green-50 text-green-700'}`}>
              {graph.cached ? 'Cached' : 'Live'}
            </span>
            <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2 py-0.5 text-gray-600">
              <Network className="h-3 w-3" /> {graph.nodes.length} nodes · {graph.edges.length} edges
            </span>
            <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2 py-0.5 text-gray-600">
              <Clock className="h-3 w-3" /> {graph.elapsed}s · effort {graph.effort}
            </span>
            {graph.spend && (
              <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2 py-0.5 text-gray-600">
                <DollarSign className="h-3 w-3" /> You.com spend ${graph.spend.total?.toFixed?.(3) ?? graph.spend.total}
              </span>
            )}
          </div>
        )}
      </Card>

      {loading && (
        <Card className="border-primary-200 bg-primary-50/40">
          <div className="flex items-center gap-3 py-4 text-primary-700">
            <Loader2 className="h-5 w-5 animate-spin" />
            <div>
              <div className="font-medium">You.com is researching “{topic}”…</div>
              <div className="text-sm text-primary-600/80">
                Multi-step agentic research with a structured graph schema. This can take
                {effort === 'standard' ? ' ~30–90s' : effort === 'deep' ? ' up to 2 min' : ' up to 5 min'}.
              </div>
            </div>
          </div>
        </Card>
      )}

      {graph && (
        <div className="grid gap-4 lg:grid-cols-3">
          {/* Graph */}
          <div className="lg:col-span-2">
            <ResearchGraph
              graph={graph}
              onNodeClick={setSelected}
              selectedId={selected?.id}
            />
            {/* Legend */}
            <div className="mt-3 flex flex-wrap gap-3">
              {(Object.keys(TYPE_META) as GraphNodeType[]).map((t) => {
                const Icon = TYPE_META[t].icon;
                const c = counts[t] || 0;
                if (!c) return null;
                return (
                  <span key={t} className="inline-flex items-center gap-1.5 text-xs text-gray-500">
                    <span className="h-2.5 w-2.5 rounded-full" style={{ background: TYPE_COLORS[t] }} />
                    <Icon className="h-3.5 w-3.5" /> {TYPE_META[t].label} ({c})
                  </span>
                );
              })}
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-4">
            {/* Selected node */}
            {selected ? (
              <Card padding="sm" className="border-primary-200">
                <div className="mb-1 flex items-center gap-2">
                  <span
                    className="rounded px-1.5 py-0.5 text-[10px] font-semibold uppercase text-white"
                    style={{ background: TYPE_COLORS[selected.type] }}
                  >
                    {selected.type}
                  </span>
                  <span className="font-semibold text-gray-900">{selected.label}</span>
                </div>
                {selected.detail && (
                  <p className="text-sm text-gray-600">{selected.detail}</p>
                )}
                {selected.url && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    <a
                      href={selected.url}
                      target="_blank"
                      rel="noreferrer"
                      className="inline-flex items-center gap-1 text-xs text-primary-600 hover:underline"
                    >
                      <ExternalLink className="h-3 w-3" /> Open source
                    </a>
                    {selected.type === 'paper' && (
                      <button
                        onClick={() => summarize(selected)}
                        disabled={compilingUrl === selected.url}
                        className="inline-flex items-center gap-1 rounded-md bg-primary-50 px-2 py-1 text-xs font-medium text-primary-700 hover:bg-primary-100 disabled:opacity-60"
                      >
                        {compilingUrl === selected.url ? (
                          <Loader2 className="h-3 w-3 animate-spin" />
                        ) : (
                          <BookOpen className="h-3 w-3" />
                        )}
                        Summarize to wiki
                      </button>
                    )}
                  </div>
                )}
              </Card>
            ) : (
              <Card padding="sm">
                <p className="text-sm text-gray-400">
                  Click any node to inspect it. Paper nodes can be sent to the Librarian.
                </p>
              </Card>
            )}

            {/* Landscape brief */}
            {graph.summary && (
              <Card padding="sm">
                <h3 className="mb-2 flex items-center gap-2 text-sm font-semibold text-gray-900">
                  <Sparkles className="h-4 w-4 text-primary-600" /> Landscape brief
                </h3>
                <div className="prose prose-sm max-w-none text-gray-600">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{graph.summary}</ReactMarkdown>
                </div>
              </Card>
            )}

            {/* Sources */}
            {graph.sources.length > 0 && (
              <Card padding="sm">
                <h3 className="mb-2 text-sm font-semibold text-gray-900">
                  Cited sources ({graph.sources.length})
                </h3>
                <div className="space-y-2">
                  {graph.sources.map((s, i) => (
                    <a
                      key={i}
                      href={s.url}
                      target="_blank"
                      rel="noreferrer"
                      className="block rounded-lg border border-gray-100 p-2 hover:border-primary-200 hover:bg-primary-50/40"
                    >
                      <div className="truncate text-xs font-medium text-gray-700">
                        {s.title || s.url}
                      </div>
                      <div className="truncate text-[11px] text-gray-400">{s.url}</div>
                    </a>
                  ))}
                </div>
              </Card>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
