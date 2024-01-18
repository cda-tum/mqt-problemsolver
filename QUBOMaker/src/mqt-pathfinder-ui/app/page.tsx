'use client'

import Image from 'next/image'
import Toggle from './toggle'
import ToggleBag from './toggleBag'
import EdgeDisplay from './edgeDisplay'
import EdgeCollector from './edgeCollector'
import GraphView from './graphView'
import { useEffect, useRef, useState } from 'react'
import TitledTextbox from './titledTextbox'
import Settings from './settings'

function download(filename: string, text: string) {
  var element = document.createElement('a');
  element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
  element.setAttribute('download', filename);

  element.style.display = 'none';
  document.body.appendChild(element);

  element.click();

  document.body.removeChild(element);
}

export default function Home() {
  const [settings, setSettings] = useState(new Settings())
  const [doUpload, setDoUpload] = useState(false);
  const [adjacencyMatrix, setAdjacencyMatrix] = useState<number[][]>([[0, 4, 2, 0],
    [3, 0, 0, 2],
    [2, 0, 0, 5],
    [2, 0, 5, 0]
    ]);

  useEffect(() => {
    if(doUpload)
      setDoUpload(false);
  }, [doUpload]);

  useEffect(() => {
    setVertices(Array.from(Array(adjacencyMatrix.length).keys()).map(i => (i + 1).toFixed(0)));

  }, [adjacencyMatrix])

  const [vertices, setVertices] = useState(Array.from(Array(adjacencyMatrix.length).keys()).map(i => (i + 1).toFixed(0)));

  return (
    <main className="flex h-screen flex-col items-center">
      <nav className="flex items-center justify-between w-full bg-slate-100 p-5">
        <a href="#" className="text-gray-900 font-medium flex flex-row gap-2 items-center"><Image alt="TUM" src={"tum_logo.svg"} width={60} height={1}></Image>MQT Pathfinder</a>
        <a href="#" className="text-gray-900 font-medium">More on our Work</a>
        <a href="#" className="text-gray-900 font-medium">Legal Information</a>
      </nav>
      <div className="flex flex-row h-full w-full items-center p-5 justify-between gap-4">
        <div className="flex flex-col gap-4">
          <GraphView updateAdjacencyMatrix={setAdjacencyMatrix} upload={doUpload} initialAdjacencyMatrix={adjacencyMatrix}></GraphView>
          <div className="flex flex-row justify-around">
            <button onClick={() => setDoUpload(true)} className="border-2 rounded bg-slate-100 p-2 hover:bg-slate-200 active:bg-slate-300">Change Graph</button>
            <button onClick={() => download("generator.json", settings.toJson())} className="border-2 rounded bg-slate-100 p-2 hover:bg-slate-200 active:bg-slate-300">Generate</button>
          </div>
          <ToggleBag cols={3} all={false} title='Encoding' items={[
            "One-Hot",
            "Domain Wall",
            "Binary",
          ]} mutualExclusions={[
            [0, 1, 2]
          ]} onChange={(states) => setSettings(settings.setEncoding(states))}></ToggleBag>
          <div className="flex flex-row gap-4">
          <TitledTextbox title="Number of Paths:" defaultValue="1" onChange={(n) => {
            const newNum = parseInt(n);
            if(!isNaN(newNum) && newNum > 0)
              setSettings(settings.setNPaths(newNum));
          }}></TitledTextbox>
          <TitledTextbox title="Max Path Length:" defaultValue={adjacencyMatrix.length + ""} onChange={(n) => {
            const newNum = parseInt(n);
            if(!isNaN(newNum) && newNum > 0)
              setSettings(settings.setMaxPathLength(newNum));
          }}></TitledTextbox>
          </div>
        </div>
        <div className="flex flex-col h-full w-4/12 justify-around">
          <ToggleBag cols={2} all={false} title='' items={[
            "Path is Loop",
            "Minimize Weight",
            "Maximize Weight",
          ]} mutualExclusions={[
            [1, 2]
          ]} onChange={(states) => setSettings(settings.setGeneralSettings(states))}></ToggleBag>
          <ToggleBag onChange={(states) => setSettings(settings.setExactlyOnceVertices(states))} all={true} title='The following vertices must appear exactly once in each path' items={vertices}></ToggleBag>
          <ToggleBag onChange={(states) => setSettings(settings.setAtLeastOnceVertices(states))} all={true} title='The following vertices must appear at least once in each path' items={vertices}></ToggleBag>
          <ToggleBag onChange={(states) => setSettings(settings.setAtMostOnceVertices(states))} all={true} title='The following vertices must appear at most once in each path' items={vertices}></ToggleBag>

          <ToggleBag onChange={(states) => setSettings(settings.setShareNoVertices(states))} all={true} title='The following paths may not share vertices' items={Array(settings.nPaths).fill(0).map((_, i) => (i + 1) + "")}></ToggleBag>
          <ToggleBag onChange={(states) => setSettings(settings.setShareNoVertices(states))} all={true} title='The following paths may not share edges' items={Array(settings.nPaths).fill(0).map((_, i) => (i + 1) + "")}></ToggleBag>
        </div>
        <div className="flex flex-col h-full w-3/12 justify-around">
          <EdgeCollector onChange={(edges) => setSettings(settings.setExactlyOnceEdges(edges))} title="The following edges must appear exactly once in each path" allVertices={vertices}></EdgeCollector>
          <EdgeCollector onChange={(edges) => setSettings(settings.setAtLeastOnceEdges(edges))} title="The following edges must appear at least once in each path" allVertices={vertices}></EdgeCollector>
          <EdgeCollector onChange={(edges) => setSettings(settings.setAtMostOnceEdges(edges))} title="The following edges must appear at most once in each path" allVertices={vertices}></EdgeCollector>


          <EdgeCollector onChange={(edges) => setSettings(settings.setPrecedence(edges))} title="The following precedence constraints must be fulfilled" allVertices={vertices}></EdgeCollector>
        </div>
      </div>
    </main>
  )
}
