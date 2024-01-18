import React, { useState } from 'react';
import Toggle from './toggle';

interface ToggleBagProps {
    title: string;
    items: string[];
    all: boolean;
    cols?: number;
    mutualExclusions?: number[][];
    onChange?: (states: boolean[]) => void;
}

const ToggleBag: React.FC<ToggleBagProps> = ({ title, items, all, cols = 4, mutualExclusions = [], onChange }) => {
    // Component logic goes here
    const [states, setStates] = useState(Array(items.length + (all ? 1 : 0)).fill(false));

    const update = (index: number, state: boolean) => {
        const newStates = [...states];
        for(const exclusion of mutualExclusions) {
            if(exclusion.includes(index)) {
                for(const i of exclusion) {
                    newStates[i + (all ? 1 : 0)] = false;
                }
            }
        }
        newStates[index + (all ? 1 : 0)] = state;
        if(index == -1 && states.every((state) => state))
            newStates.fill(false);
        else if(index == -1)
            newStates.fill(true);
        else if(all && newStates.every((state, index) => index == 0 || state))
            newStates[0] = true;
        else if(all && newStates.every((state, index) => index == 0 || !state))
            newStates[0] = false;
        setStates(newStates);
        onChange?.(newStates);
    }

    return (
        // JSX markup goes here
        <div>
            <h1>{title}</h1>
            <div className={`grid grid-cols-${cols} gap-4`}>
                {
                    (all ? (["ALL"].concat(items)) : items).map((item, index) => (
                        <Toggle state={states[index]} onUpdate={(state) => update(index - (all ? 1 : 0), state)} key={index}>{item}</Toggle>
                    ))
                }
            </div>
        </div>
    );
};

export default ToggleBag;
