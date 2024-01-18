import React, { useState } from 'react';
import Toggle from './toggle';
import EdgeDisplay from './edgeDisplay';

interface EdgeCollectorProps {
    title: string;
    cols?: number;
    allVertices: string[];
    onChange?: (edges: string[][]) => void;
}

const EdgeCollector: React.FC<EdgeCollectorProps> = ({ title, cols = 4, allVertices, onChange}) => {
    // Component logic goes here
    const [items, setItems] = useState<string[][]>([]);

    const itemAdded = (fromVertex: string, toVertex: string) => {
        if(fromVertex === "" || toVertex === "")
            return;
        if(items.some((item) => item[0] == fromVertex && item[1] == toVertex))
            return;
        const newItems = [...items];
        newItems.push([fromVertex, toVertex]);
        setItems(newItems);
        onChange?.(newItems);
    }

    const itemRemoved = (fromVertex: string, toVertex: string) => {
        const newItems = [...items];
        newItems.splice(newItems.findIndex((item) => item[0] == fromVertex && item[1] == toVertex), 1);
        setItems(newItems);
        onChange?.(newItems);
    }

    const fromRef = React.createRef<HTMLSelectElement>();
    const toRef = React.createRef<HTMLSelectElement>();

    return (
        // JSX markup goes here
        <div>
            <h1>{title}</h1>
            <div className='flex flex-row gap-4 mb-2'>
                <select ref={fromRef} className="border-2 rounded p-1">
                    {
                        allVertices.map((vertex, index) => (
                            <option key={index}>{vertex}</option>
                        ))
                    }
                </select>
                <span className="pb-1 pt-1">➔</span>
                <select ref={toRef} className="border-2 rounded p-1">
                    {
                        allVertices.map((vertex, index) => (
                            <option key={index}>{vertex}</option>
                        ))
                    }
                </select>
                <button className='border-2 rounded bg-slate-100 p-1 hover:bg-slate-200 active:bg-slate-300'
                    onClick={() => itemAdded(fromRef.current?.value ?? "", toRef.current?.value ?? "")}>Add</button>
            </div>
            <div className={`grid grid-cols-${cols} gap-4`}>
                {
                    items.map((item, index) => (
                        <EdgeDisplay
                            onClick={(fromVertex, toVertex) => itemRemoved?.(fromVertex, toVertex)}
                            fromVertex={item[0]}
                            toVertex={item[1]}
                            key={index}>
                        </EdgeDisplay>
                    ))
                }
            </div>
        </div>
    );
};

export default EdgeCollector;
