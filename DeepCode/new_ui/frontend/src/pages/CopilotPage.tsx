import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import {
  Search, BookOpen, Compass, Wrench, Loader2, CheckCircle,
  XCircle, ArrowRight, FileText, Sparkles, Network,
} from 'lucide-react';
import { Card, Button } from '../components/common';
import { ProgressTracker, ActivityLogViewer } from '../components/streaming';
import { FileTree } from '../components/results';
import { InteractionPanel } from '../components/interaction';
import { useWorkflowStore } from '../stores/workflowStore';
import { useSessionStore } from '../stores/sessionStore';
import { useStreaming } from '../hooks/useStreaming';
import { copilotApi, workflowsApi } from '../services/api';
import type { PaperHit, CompileResponse, PrebakedResponse } from '../services/api';
import { toast } from '../components/common/Toaster';
import { PAPER_TO_CODE_STEPS } from '../types/workflow';
import type { WorkflowStep } from '../types/workflow';

type Stage = 'interest' | 'papers' | 'wiki' | 'reproduce';

interface DemoLog {
  id: string;
  timestamp: Date;
  message: string;
  progress: number;
  type: 'info' | 'success' | 'warning' | 'error' | 'progress';
}

const DEMO_STEP_MESSAGES: Record<string, string> = {
  init: 'Booting Architect + Engineer…',
  input: 'Fetched paper PDF, converting to markdown…',
  workspace: 'Prepared task workspace…',
  preprocess: 'Segmented paper into sections…',
  planning: 'Drafted implementation plan (MSCP + binning + NDCG eval)…',
  references: 'Analyzed related retrieval / reranking work…',
  implementation: 'Generating code files…',
};

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

const AGENTS = [
  { key: 'scout', name: 'Scout', icon: Search, blurb: 'Discovers papers via You.com' },
  { key: 'librarian', name: 'Librarian', icon: BookOpen, blurb: 'Compiles to your wiki' },
  { key: 'architect', name: 'Architect', icon: Compass, blurb: 'Clarifies & plans' },
  { key: 'engineer', name: 'Engineer', icon: Wrench, blurb: 'Reproduces the code' },
];

export default function CopilotPage() {
  const navigate = useNavigate();
  const [stage, setStage] = useState<Stage>('interest');
  const [interest, setInterest] = useState('LLM confidence reranker for RAG');
  const [discovering, setDiscovering] = useState(false);
  const [papers, setPapers] = useState<PaperHit[]>([]);
  const [selected, setSelected] = useState<PaperHit | null>(null);
  const [compiling, setCompiling] = useState(false);
  const [wiki, setWiki] = useState<CompileResponse | null>(null);

  // Demo mode: play a streaming veneer, then render the pre-baked repo (ADR-0002)
  const [demoMode, setDemoMode] = useState(true);
  const [prebaked, setPrebaked] = useState<PrebakedResponse | null>(null);
  const [demoSteps, setDemoSteps] = useState<WorkflowStep[]>(PAPER_TO_CODE_STEPS);
  const [demoProgress, setDemoProgress] = useState(0);
  const [demoMessage, setDemoMessage] = useState('');
  const [demoLogs, setDemoLogs] = useState<DemoLog[]>([]);
  const [demoRunning, setDemoRunning] = useState(false);
  const [selectedFile, setSelectedFile] = useState<string | undefined>(undefined);
  const [fileContent, setFileContent] = useState('');
  const [fileLoading, setFileLoading] = useState(false);

  const {
    activeTaskId, status, progress, message, steps, generatedFiles,
    activityLogs, pendingInteraction, isWaitingForInput, result, error,
    setActiveTask, setSteps, reset,
  } = useWorkflowStore();
  const { activeSessionId, setActiveSessionId, selectSession } = useSessionStore();

  useStreaming(activeTaskId);

  const activeAgent =
    stage === 'interest' || stage === 'papers' ? 'scout'
      : stage === 'wiki' ? 'librarian'
        : isWaitingForInput ? 'architect' : 'engineer';

  const handleDiscover = async () => {
    if (!interest.trim()) return;
    setDiscovering(true);
    try {
      const res = await copilotApi.discover(interest.trim());
      // de-dupe by arxiv_id / url
      const seen = new Set<string>();
      const unique = res.papers.filter((p) => {
        const k = p.arxiv_id || p.url;
        if (seen.has(k)) return false;
        seen.add(k);
        return true;
      });
      setPapers(unique);
      setStage('papers');
      if (!unique.length) toast.info('No papers found', 'Try a different interest.');
    } catch (err) {
      toast.error('Scout failed', 'Could not reach You.com search.');
      console.error(err);
    } finally {
      setDiscovering(false);
    }
  };

  const handleSelect = async (paper: PaperHit) => {
    setSelected(paper);
    setCompiling(true);
    setStage('wiki');
    try {
      const res = await copilotApi.compileWiki(paper.url, 'llm-agents', interest);
      setWiki(res);
      toast.success('Added to wiki', res.rel_path);
    } catch (err) {
      toast.error('Librarian failed', 'Wiki compile did not complete.');
      console.error(err);
    } finally {
      setCompiling(false);
    }
  };

  const handleReproduce = async () => {
    if (!selected) return;
    try {
      reset();
      setSteps(PAPER_TO_CODE_STEPS);
      setStage('reproduce');
      const res = await workflowsApi.startPaperToCode(
        selected.url, 'url', false, true, activeSessionId
      );
      setActiveTask(res.task_id, 'paper-to-code');
      if (res.session_id) {
        setActiveSessionId(res.session_id);
        selectSession(res.session_id);
      }
      toast.info('Reproduction started', 'Architect and Engineer are on it.');
    } catch (err) {
      toast.error('Failed to start reproduction', 'Please try again.');
      console.error(err);
    }
  };

  const handleDemoReproduce = async () => {
    if (!selected) return;
    setStage('reproduce');
    setDemoRunning(true);
    setDemoLogs([]);
    setDemoProgress(0);
    setPrebaked(null);
    setSelectedFile(undefined);
    setFileContent('');

    let steps: WorkflowStep[] = PAPER_TO_CODE_STEPS.map((s) => ({ ...s, status: 'pending' }));
    setDemoSteps(steps);

    // Fetch the cached repo in the background while the veneer plays.
    const prebakedPromise = copilotApi.getPrebaked().catch((err) => {
      console.error(err);
      return null;
    });

    for (let i = 0; i < steps.length; i++) {
      steps = steps.map((s, idx) => ({
        ...s,
        status: idx < i ? 'completed' : idx === i ? 'active' : 'pending',
      })) as WorkflowStep[];
      setDemoSteps([...steps]);
      const step = steps[i];
      const msg = DEMO_STEP_MESSAGES[step.id] || step.title;
      setDemoProgress(step.progress);
      setDemoMessage(msg);
      setDemoLogs((prev) => [
        ...prev,
        { id: `${step.id}-${Date.now()}`, timestamp: new Date(), message: msg, progress: step.progress, type: 'progress' },
      ]);
      await sleep(850 + Math.random() * 650);
    }

    const data = await prebakedPromise;
    steps = steps.map((s) => ({ ...s, status: 'completed' })) as WorkflowStep[];
    setDemoSteps([...steps]);
    setDemoProgress(100);
    if (data) {
      setPrebaked(data);
      const done = `Reproduction complete: ${data.file_count} files in ${data.repo_name}`;
      setDemoMessage(done);
      setDemoLogs((prev) => [
        ...prev,
        { id: `done-${Date.now()}`, timestamp: new Date(), message: done, progress: 100, type: 'success' },
      ]);
    } else {
      setDemoMessage('Could not load pre-baked reproduction. Is the backend up?');
      setDemoLogs((prev) => [
        ...prev,
        { id: `err-${Date.now()}`, timestamp: new Date(), message: 'No pre-baked reproduction found.', progress: 100, type: 'error' },
      ]);
    }
    setDemoRunning(false);
  };

  const handleFileSelect = async (path: string) => {
    setSelectedFile(path);
    setFileLoading(true);
    try {
      const res = await copilotApi.getPrebakedFile(path);
      setFileContent(res.content);
    } catch (err) {
      setFileContent('// Failed to load file.');
      console.error(err);
    } finally {
      setFileLoading(false);
    }
  };

  const restart = () => {
    reset();
    setStage('interest');
    setPapers([]);
    setSelected(null);
    setWiki(null);
    setPrebaked(null);
    setDemoRunning(false);
    setDemoLogs([]);
    setSelectedFile(undefined);
    setFileContent('');
  };

  const isRunning = status === 'running';

  return (
    <div className="space-y-6">
      <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex items-center gap-2">
          <Sparkles className="h-6 w-6 text-primary-600" />
          <h1 className="text-2xl font-bold text-gray-900">Research Copilot</h1>
        </div>
        <p className="text-gray-500 mt-1">
          Discover a paper, summarize it to your wiki, then reproduce it as running code.
        </p>
      </motion.div>

      {/* Agent team rail */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {AGENTS.map((a) => {
          const Icon = a.icon;
          const isActive = a.key === activeAgent;
          return (
            <Card key={a.key} padding="sm"
              className={isActive ? 'border-primary-300 bg-primary-50' : ''}>
              <div className="flex items-center gap-2">
                <Icon className={`h-5 w-5 ${isActive ? 'text-primary-600' : 'text-gray-400'}`} />
                <div>
                  <div className={`text-sm font-semibold ${isActive ? 'text-primary-700' : 'text-gray-700'}`}>
                    {a.name}
                  </div>
                  <div className="text-[11px] text-gray-400">{a.blurb}</div>
                </div>
              </div>
            </Card>
          );
        })}
      </div>

      {/* Stage: interest */}
      {stage === 'interest' && (
        <Card>
          <h3 className="font-semibold text-gray-900 mb-3">What are you researching?</h3>
          <div className="flex flex-col gap-3 sm:flex-row">
            <input
              value={interest}
              onChange={(e) => setInterest(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleDiscover()}
              placeholder="e.g. LLM confidence reranker for RAG"
              className="flex-1 rounded-lg border border-gray-300 px-4 py-2 text-sm focus:border-primary-500 focus:outline-none focus:ring-1 focus:ring-primary-500"
            />
            <Button onClick={handleDiscover} isLoading={discovering}>
              <Search className="mr-2 h-4 w-4" /> Send Scout
            </Button>
          </div>
        </Card>
      )}

      {/* Stage: papers */}
      {stage === 'papers' && (
        <Card>
          <div className="mb-3 flex items-center justify-between">
            <h3 className="font-semibold text-gray-900">
              Scout found {papers.length} paper{papers.length === 1 ? '' : 's'}
            </h3>
            <div className="flex items-center gap-3">
              <button
                onClick={() => navigate(`/knowledge-graph?topic=${encodeURIComponent(interest)}`)}
                className="inline-flex items-center gap-1 text-xs font-medium text-primary-600 hover:text-primary-700"
              >
                <Network className="h-3.5 w-3.5" /> Map the landscape
              </button>
              <button onClick={restart} className="text-xs text-gray-400 hover:text-gray-600">
                New search
              </button>
            </div>
          </div>
          <div className="space-y-3">
            {papers.map((p) => (
              <motion.div key={p.url} initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <div className="group flex items-start justify-between gap-4 rounded-lg border border-gray-200 p-4 hover:border-primary-300 hover:bg-primary-50/40">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <FileText className="h-4 w-4 flex-shrink-0 text-gray-400" />
                      <span className="truncate font-medium text-gray-900">{p.title}</span>
                    </div>
                    <p className="mt-1 line-clamp-2 text-xs text-gray-500">{p.snippet}</p>
                    <div className="mt-1 text-[11px] text-gray-400">
                      {p.arxiv_id && <span className="mr-2 font-mono">arXiv:{p.arxiv_id}</span>}
                      {p.published?.slice(0, 10)}
                    </div>
                  </div>
                  <Button size="sm" variant="secondary" onClick={() => handleSelect(p)}>
                    Summarize <ArrowRight className="ml-1 h-3 w-3" />
                  </Button>
                </div>
              </motion.div>
            ))}
          </div>
        </Card>
      )}

      {/* Stage: wiki */}
      {(stage === 'wiki' || stage === 'reproduce') && selected && (
        <Card>
          <div className="mb-3 flex items-center justify-between">
            <h3 className="font-semibold text-gray-900 flex items-center gap-2">
              <BookOpen className="h-4 w-4 text-primary-600" /> Wiki article
            </h3>
            {wiki && (
              <span className="rounded-full bg-gray-100 px-2 py-0.5 font-mono text-[11px] text-gray-500">
                wiki/{wiki.rel_path}
              </span>
            )}
          </div>
          {compiling ? (
            <div className="flex items-center gap-2 py-8 text-gray-500">
              <Loader2 className="h-5 w-5 animate-spin" /> Librarian is compiling…
            </div>
          ) : wiki ? (
            <>
              <div className="prose prose-sm max-w-none rounded-lg border border-gray-100 bg-gray-50 p-4">
                <ReactMarkdown>{wiki.markdown}</ReactMarkdown>
              </div>
              {stage === 'wiki' && (
                <div className="mt-4 flex items-center justify-end gap-4">
                  <label className="flex cursor-pointer items-center gap-2 text-xs text-gray-500">
                    <input
                      type="checkbox"
                      checked={demoMode}
                      onChange={(e) => setDemoMode(e.target.checked)}
                      className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
                    />
                    Instant demo (pre-baked)
                  </label>
                  <Button onClick={demoMode ? handleDemoReproduce : handleReproduce}>
                    <Wrench className="mr-2 h-4 w-4" /> Reproduce this paper
                  </Button>
                </div>
              )}
            </>
          ) : (
            <p className="py-6 text-sm text-gray-400">Waiting for Librarian…</p>
          )}
        </Card>
      )}

      {/* Stage: reproduce (demo mode — pre-baked, per ADR-0002) */}
      {stage === 'reproduce' && demoMode && (
        <div className="space-y-6">
          <Card>
            <ProgressTracker steps={demoSteps} currentProgress={demoProgress} currentMessage={demoMessage} />
          </Card>

          <ActivityLogViewer logs={demoLogs} isRunning={demoRunning} currentMessage={demoMessage} />

          {prebaked && (
            <>
              <Card className="border-green-200 bg-green-50">
                <div className="flex items-start gap-3">
                  <CheckCircle className="h-6 w-6 flex-shrink-0 text-green-500" />
                  <div>
                    <h3 className="font-medium text-green-900">
                      Reproduced <span className="font-mono">{prebaked.repo_name}</span> — {prebaked.file_count} files
                    </h3>
                    <p className="mt-1 text-sm text-green-700">
                      Training-free LCR reranker (MSCP confidence + binning), with BM25/Contriever
                      retrievers and an NDCG@5 eval harness. Run it with{' '}
                      <span className="font-mono">python run.py</span> (needs a 7–9B LLM + a BEIR set
                      like NFCorpus).
                    </p>
                  </div>
                </div>
              </Card>

              <div className="grid gap-4 lg:grid-cols-2">
                <FileTree
                  files={prebaked.files.map((f) => f.path)}
                  onFileSelect={handleFileSelect}
                  selectedFile={selectedFile}
                />
                <Card padding="sm">
                  <div className="mb-2 flex items-center gap-2 border-b border-gray-100 pb-2">
                    <FileText className="h-4 w-4 text-gray-400" />
                    <span className="truncate font-mono text-xs text-gray-600">
                      {selectedFile || 'Select a file to view'}
                    </span>
                  </div>
                  {fileLoading ? (
                    <div className="flex items-center gap-2 py-8 text-gray-500">
                      <Loader2 className="h-4 w-4 animate-spin" /> Loading…
                    </div>
                  ) : selectedFile ? (
                    <pre className="max-h-[400px] overflow-auto rounded bg-gray-900 p-3 text-[11px] leading-relaxed text-gray-100">
                      <code>{fileContent}</code>
                    </pre>
                  ) : (
                    <p className="py-8 text-center text-sm text-gray-400">
                      Click a file in the tree to preview its contents.
                    </p>
                  )}
                </Card>
              </div>
            </>
          )}

          <div className="flex justify-start">
            <button onClick={restart} className="text-xs text-gray-400 hover:text-gray-600">
              Start over
            </button>
          </div>
        </div>
      )}

      {/* Stage: reproduce (live mode) */}
      {stage === 'reproduce' && !demoMode && (
        <div className="space-y-6">
          {status !== 'idle' && (
            <Card>
              <ProgressTracker steps={steps} currentProgress={progress} currentMessage={message} />
            </Card>
          )}

          <AnimatePresence>
            {pendingInteraction && activeTaskId && (
              <InteractionPanel taskId={activeTaskId} interaction={pendingInteraction} />
            )}
          </AnimatePresence>

          <ActivityLogViewer
            logs={activityLogs}
            isRunning={isRunning && !isWaitingForInput}
            currentMessage={isWaitingForInput ? 'Architect is waiting for your input…' : message}
          />

          {generatedFiles.length > 0 && <FileTree files={generatedFiles} />}

          {status === 'completed' && result && (
            <Card className="border-green-200 bg-green-50">
              <div className="flex items-start gap-3">
                <CheckCircle className="h-6 w-6 flex-shrink-0 text-green-500" />
                <div>
                  <h3 className="font-medium text-green-900">Reproduction complete</h3>
                  <p className="mt-1 text-sm text-green-700">
                    The Engineer generated a working implementation of the paper.
                  </p>
                </div>
              </div>
            </Card>
          )}

          {status === 'error' && error && (
            <Card className="border-red-200 bg-red-50">
              <div className="flex items-start gap-3">
                <XCircle className="h-6 w-6 flex-shrink-0 text-red-500" />
                <div>
                  <h3 className="font-medium text-red-900">Reproduction failed</h3>
                  <p className="mt-1 text-sm text-red-700">{error}</p>
                </div>
              </div>
            </Card>
          )}

          <div className="flex justify-start">
            <button onClick={restart} className="text-xs text-gray-400 hover:text-gray-600">
              Start over
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
