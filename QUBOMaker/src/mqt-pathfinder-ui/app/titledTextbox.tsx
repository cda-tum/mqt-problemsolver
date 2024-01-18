import React from 'react';

type TitledTextboxProps = {
  title: string;
  defaultValue?: string;
  onChange?: (value: string) => void;
};

const TitledTextbox: React.FC<TitledTextboxProps> = ({ title, defaultValue, onChange }) => {
  return (
    <div className="flex flex-col flex-1">
      <label className="p-1 mr-4">{title}</label>
      <input className="border-2 rounded p-1" type="text" defaultValue={defaultValue} onChange={(e) => onChange?.(e.currentTarget.value!)} />
    </div>
  );
};

export default TitledTextbox;
