
import React from 'react';

interface EdgeDisplayProps {
    fromVertex: string;
    toVertex: string;
    onClick?: (fromVertex: string, toVertex: string) => void;
}

const EdgeDisplay: React.FC<EdgeDisplayProps> = ({ fromVertex, toVertex, onClick }) => {
    return (
        <div onClick={() => onClick?.(fromVertex, toVertex)} className={`text-center border-2 rounded p-1 ${onClick !== undefined ? "cursor-pointer" : ""}`}>
            <span>{fromVertex}</span>
            <span className="ml-3 mr-3">➔</span>
            <span>{toVertex}</span>
        </div>
    );
};

export default EdgeDisplay;
