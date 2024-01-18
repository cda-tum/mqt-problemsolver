import React, { useState, useRef, useEffect } from 'react';
import styles from './style.module.css'
import Script from 'next/script';
import Graph from 'react-graph-vis';

const stringToAdjacencyMatrix = (str: string): number[][] | undefined => {
    const lines = str.trim().split('\n');
    const matrix: number[][] = [];
    for(const line of lines) {
        const row: number[] = [];
        for(const char of line.split(" ")) {
            const result = parseFloat(char);
            if(isNaN(result))
                return undefined;
            row.push(result);
        }
        matrix.push(row);
    }
    return matrix;
}

const adjacencyMatrixToString = (matrix: number[][]): string => {
    let str = "";
    for(const row of matrix) {
        for(const cell of row) {
            str += cell + " ";
        }
        str = str.trimEnd();
        str += "\n";
    }
    return str;
}

const adjacencyMatrixToDict = (matrix: number[][]): { [key: string]: any } => {
    const dict: { [key: string]: any } = {};
    dict["nodes"] = [];
    dict["edges"] = [];
    for(let i = 0; i < matrix.length; i++) {
        dict["nodes"].push({"id": i + 1, "label": "" + (i + 1)});
    }
    for(let i = 0; i < matrix.length; i++) {
        for(let j = 0; j < matrix[i].length; j++) {
            if(matrix[i][j] > 0)
                dict["edges"].push({"from": i + 1, "to": j + 1, "label": ""+matrix[i][j], "id": `${i + 1}-to-${j + 1}`});
        }
    }
    return dict;
}

interface GraphViewProps {
    upload: boolean,
    updateAdjacencyMatrix: (adjacencyMatrix: number[][]) => void,
    initialAdjacencyMatrix?: number[][]
}


const GraphView: React.FC<GraphViewProps> = ({ upload, updateAdjacencyMatrix, initialAdjacencyMatrix }) => {
    const [adjacencyMatrix, setAdjacencyMatrix] = useState<number[][]>(initialAdjacencyMatrix ?? []);
    const [uploadMode, setUploadMode] = useState<boolean>(false);
    const textareaRef = useRef<HTMLTextAreaElement>(null);
    const [isClient, setIsClient] = useState(false);

    useEffect(() => {
        setIsClient(true)
    }, []);

    if(upload && !uploadMode) {
        console.log(adjacencyMatrix);
        setUploadMode(true);
    }

    const doOk = () => {
        const textareaValue = textareaRef.current?.value;
        const adj = stringToAdjacencyMatrix(textareaValue ?? "");
        if(adj !== undefined) {
            console.log(adj);
            setAdjacencyMatrix(adj);
            updateAdjacencyMatrix(adj);
        }
        setUploadMode(false);
    }

    const doCancel = () => {
        textareaRef.current!.value = adjacencyMatrixToString(adjacencyMatrix);
        setUploadMode(false);
    }

    const graph = adjacencyMatrixToDict(adjacencyMatrix);
    const options = {
        layout: {
          hierarchical: false,
        },
        nodes: {
            color: {
                background: "white",
                border: "black"
            },
        },
        edges: {
          arrows: {
            to: { enabled: true, scaleFactor: 1, type: 'arrow',  },
          },
          smooth: true,
          font: {
            align: "top"
          }
        },
      };

    return (
        <div className={styles.canvas}>
                {
                        uploadMode || !isClient ? (
                                <div className="w-full h-full flex flex-col gap-2">
                                        <textarea defaultValue={adjacencyMatrixToString(adjacencyMatrix)} ref={textareaRef} className="w-full p-2 flex-1 font-mono"></textarea>
                                        <div className="w-full flex flex-row justify-around mb-1">
                                                <button onClick={doCancel} className="border-2 rounded bg-slate-100 p-2 hover:bg-slate-200 active:bg-slate-300">Cancel</button>
                                                <button onClick={doOk} className="border-2 rounded bg-slate-100 p-2 hover:bg-slate-200 active:bg-slate-300">OK</button>
                                        </div>
                                </div>
                        ) : (
                            <Graph className="w-full h-full" graph={graph} options={options}/>
                        )
                }
        </div>
    )
};

export default GraphView;
