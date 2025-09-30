'use client';

import { useState, useEffect, useRef } from 'react';
import { useParams, useRouter, useSearchParams } from 'next/navigation';
import Image from 'next/image';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { ArrowLeft, Save, RotateCcw } from 'lucide-react';
import { useToast } from '@/lib/use-toast';

interface Detection {
    class: string;
    confidence: number;
    bbox: [number, number, number, number];
}

interface ImageData {
    filename: string;
    max_confidence: number;
    detections: Detection[];
}

export default function EditPage() {
    const params = useParams();
    const router = useRouter();
    const searchParams = useSearchParams();
    const { toast } = useToast();
    const filename = params.filename as string;
    const debug = searchParams.get('debug') === 'true';
    const [imageData, setImageData] = useState<ImageData | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [editedDetections, setEditedDetections] = useState<Detection[]>([]);
    const [orignalDetections, setOriginalDetections] = useState<Detection[]>([]);
    const svgRef = useRef<SVGSVGElement>(null);
    const [dragging, setDragging] = useState<{ index: number; handle: string } | null>(null);
    const [initialPos, setInitialPos] = useState({ x: 0, y: 0 });
    const colors = ['red', 'blue', 'green', 'orange', 'purple', 'pink'];

    useEffect(() => {
        fetchImageData();
    }, [filename]);

    const fetchImageData = async () => {
        try {
            setLoading(true);
            const response = await fetch('http://localhost:5000/discover_low_confidence');
            if (!response.ok) throw new Error('Failed to fetch data');
            const data = await response.json();
            const img = data.low_confidence_images.find((i: ImageData) => i.filename === filename);
            if (!img) throw new Error('Image not found');
            setImageData(img);
            // Sort detections by confidence descending, so index 0 is most confident
            const sortedDetections = [...img.detections].sort((a, b) => b.confidence - a.confidence);
            setEditedDetections(sortedDetections);
            setOriginalDetections(JSON.parse(JSON.stringify(sortedDetections))); // Deep copy
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Error');
        } finally {
            setLoading(false);
        }
    };

    const handleMouseDown = (e: React.MouseEvent, index: number, handle: string) => {
        setDragging({ index, handle });
        const rect = svgRef.current?.getBoundingClientRect();
        if (rect) {
            setInitialPos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
        }
    };

    const handleMouseMove = (e: React.MouseEvent) => {
        if (!dragging || !svgRef.current) return;
        const rect = svgRef.current.getBoundingClientRect();
        const currentX = e.clientX - rect.left;
        const currentY = e.clientY - rect.top;
        const deltaX = (currentX - initialPos.x) / rect.width * 0.5;
        const deltaY = (currentY - initialPos.y) / rect.height * 0.5;

        setEditedDetections(prev => {
            const newDets = [...prev];
            const det = newDets[dragging.index];
            const [x, y, w, h] = det.bbox;

            const left = x - w / 2;
            const right = x + w / 2;
            const top = y - h / 2;
            const bottom = y + h / 2;
            const minSize = 0.025;

            let newLeft = left;
            let newRight = right;
            let newTop = top;
            let newBottom = bottom;

            switch (dragging.handle) {
                case 'move':
                    newLeft = Math.max(0, Math.min(1 - w, left + deltaX));
                    newTop = Math.max(0, Math.min(1 - h, top + deltaY));
                    newRight = newLeft + w;
                    newBottom = newTop + h;
                    break;
                case 'nw':
                    newLeft = Math.max(0, Math.min(right - minSize, left + deltaX));
                    newTop = Math.max(0, Math.min(bottom - minSize, top + deltaY));
                    break;
                case 'ne':
                    newRight = Math.min(1, Math.max(left + minSize, right + deltaX));
                    newTop = Math.max(0, Math.min(bottom - minSize, top + deltaY));
                    break;
                case 'sw':
                    newLeft = Math.max(0, Math.min(right - minSize, left + deltaX));
                    newBottom = Math.min(1, Math.max(top + minSize, bottom + deltaY));
                    break;
                case 'se':
                    newRight = Math.min(1, Math.max(left + minSize, right + deltaX));
                    newBottom = Math.min(1, Math.max(top + minSize, bottom + deltaY));
                    break;
            }

            const newW = newRight - newLeft;
            const newH = newBottom - newTop;
            const newX = (newLeft + newRight) / 2;
            const newY = (newTop + newBottom) / 2;

            newDets[dragging.index].bbox = [
                newX,
                newY,
                newW,
                newH
            ];
            return newDets;
        });

        setInitialPos({ x: currentX, y: currentY });
    };

    const handleMouseUp = () => {
        setDragging(null);
    };

    const saveChanges = async () => {
        try {
            // Only save the most confident (editable) detection
            const detectionToSave = editedDetections[0];
            const response = await fetch('http://localhost:5000/save_labels', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename, detections: [detectionToSave] })
            });
            if (!response.ok) throw new Error('Failed to save');
            toast({
                title: "Success",
                description: "Labels saved successfully!",
            });
        } catch (err) {
            console.error(err);
            toast({
                title: "Error",
                description: "Failed to save labels.",
                variant: "destructive",
            });
        }
    };

    const resetChanges = () => {
        console.log('Reset button clicked');
        if (orignalDetections) {
            console.log('Image data exists');
            console.log('Sorted original:', orignalDetections);
            setEditedDetections(prev => {
                const newDets = [...prev];
                console.log('Before reset:', newDets[0]);
                newDets[0] = orignalDetections[0]; // Reset only the most confident (editable) detection
                console.log('After reset:', newDets[0]);
                return newDets;
            });
            // This is a bit of a hack to ensure that originalDetections remains unchanged
            setOriginalDetections(JSON.parse(JSON.stringify(orignalDetections))); // Deep copy
        } else {
            console.log('No image data');
        }
    };

    if (loading) return <div>Loading...</div>;
    if (error) return <div>Error: {error}</div>;
    if (!imageData) return <div>Image not found</div>;

    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-gray-900 dark:to-gray-800 p-4 md:p-8">
            <div className="max-w-4xl mx-auto">
                <Card className="mb-8">
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Button variant="outline" size="sm" onClick={() => router.back()}>
                                <ArrowLeft className="w-4 h-4" />
                            </Button>
                            Edit Bounding Boxes: {filename}
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="flex gap-4">
                        <Button onClick={saveChanges} className="flex items-center gap-2">
                            <Save className="w-4 h-4" />
                            Save Changes
                        </Button>
                        <Button variant="outline" onClick={resetChanges} className="flex items-center gap-2">
                            <RotateCcw className="w-4 h-4" />
                            Reset
                        </Button>
                    </CardContent>
                </Card>

                <Card>
                    <CardContent className="p-0">
                        <div className="aspect-square relative bg-gray-100 dark:bg-gray-800">
                            <Image
                                src={`http://localhost:5000/images/${filename}`}
                                alt={filename}
                                fill
                                className="object-cover"
                            />
                            <svg
                                ref={svgRef}
                                className="absolute inset-0 w-full h-full"
                                viewBox="0 0 100 100"
                                preserveAspectRatio="none"
                                onMouseMove={handleMouseMove}
                                onMouseUp={handleMouseUp}
                                onMouseLeave={handleMouseUp}
                            >
                                {editedDetections.map((detection, index) => {
                                    const [x_center, y_center, width, height] = detection.bbox;
                                    const x = (x_center - width / 2) * 100;
                                    const y = (y_center - height / 2) * 100;
                                    const w = width * 100;
                                    const h = height * 100;
                                    const color = colors[index % colors.length];
                                    const isEditable = index === 0; // Only most confident is editable
                                    return (
                                        <g key={index}>
                                            <rect
                                                x={x}
                                                y={y}
                                                width={w}
                                                height={h}
                                                fill="none"
                                                stroke={color}
                                                strokeWidth="0.5"
                                                onMouseDown={isEditable ? (e) => handleMouseDown(e, index, 'move') : undefined}
                                                style={{ cursor: isEditable ? 'move' : 'default' }}
                                            />
                                            {isEditable && (
                                                <>
                                                    {/* Resize handles */}
                                                    <circle cx={x} cy={y} r="1" fill={color} onMouseDown={(e) => handleMouseDown(e, index, 'nw')} style={{ cursor: 'nw-resize' }} />
                                                    <circle cx={x + w} cy={y} r="1" fill={color} onMouseDown={(e) => handleMouseDown(e, index, 'ne')} style={{ cursor: 'ne-resize' }} />
                                                    <circle cx={x} cy={y + h} r="1" fill={color} onMouseDown={(e) => handleMouseDown(e, index, 'sw')} style={{ cursor: 'sw-resize' }} />
                                                    <circle cx={x + w} cy={y + h} r="1" fill={color} onMouseDown={(e) => handleMouseDown(e, index, 'se')} style={{ cursor: 'se-resize' }} />
                                                    {debug && (
                                                        <>
                                                            <text x={x - 4} y={y - 2} fontSize="3" fill="white" stroke="black" strokeWidth="0.2">NW</text>
                                                            <text x={x + w + 2} y={y - 2} fontSize="3" fill="white" stroke="black" strokeWidth="0.2">NE</text>
                                                            <text x={x - 4} y={y + h + 4} fontSize="3" fill="white" stroke="black" strokeWidth="0.2">SW</text>
                                                            <text x={x + w + 2} y={y + h + 4} fontSize="3" fill="white" stroke="black" strokeWidth="0.2">SE</text>
                                                        </>
                                                    )}
                                                </>
                                            )}
                                        </g>
                                    );
                                })}
                            </svg>
                        </div>
                    </CardContent>
                </Card>

                <Card className="mt-8">
                    <CardHeader>
                        <CardTitle>Detections</CardTitle>
                    </CardHeader>
                    <CardContent>
                        {editedDetections.map((det, index) => {
                            const color = colors[index % colors.length];
                            const isEditable = index === 0;
                            return (
                                <div key={index} className="mb-2 flex items-center">
                                    <div
                                        className="w-4 h-4 rounded mr-2"
                                        style={{ backgroundColor: color }}
                                    ></div>
                                    <p>
                                        {det.class}: {(det.confidence * 100).toFixed(1)}% {isEditable ? '(Editable)' : ''} - Bbox: {det.bbox.map(v => v.toFixed(3)).join(', ')}
                                    </p>
                                </div>
                            );
                        })}
                    </CardContent>
                </Card>
            </div>
        </div>
    );
}
